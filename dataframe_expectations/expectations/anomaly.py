"""Statistically-significant anomaly detection for a numeric column.

This module exposes ``ExpectationColumnValuesNotAnomalous`` (suite method
``expect_column_values_not_anomalous``), a column expectation that flags rows
whose value in a numeric column is a *statistically-significant* anomaly.

Adapted (Mode 2) from:
    Sugiyama, Ogawa, Hashimoto & Takeuchi (2025),
    "Statistical Inference for Clustering-based Anomaly Detection" (SI-CLAD),
    arXiv:2504.18633.

Core mechanism kept at full fidelity:
    * **Detection** -- DBSCAN clustering; points that belong to no cluster
      (noise) are the candidate anomalies.
    * **Test statistic** -- for each candidate ``j`` the contrast
      ``T_j = X_j - mean(X_inliers)`` (deviation from the non-anomaly mean),
      exactly the statistic used in the paper.
    * **Selective p-value** -- the inference conditions on the DBSCAN
      *selection event* (the observed set of anomalies). A parametric line
      search over the contrast direction identifies the values of the
      statistic that are consistent with that selection event; the p-value is
      the tail of a Gaussian truncated to that set.
    * **False-detection control** -- candidate ``j`` is flagged when its
      selective p-value ``<= alpha``. Under the pointwise null this gives a
      per-anomaly false-detection probability of exactly ``alpha`` (the
      paper's validity guarantee), instead of the uncontrolled error rate of
      naively trusting every DBSCAN noise point.

Target-native adaptations (clearly scoped, not shortfalls):
    * Univariate -- the method is applied to a single numeric column. The
      selective-inference machinery is dimension-agnostic; the multivariate
      statistic is a natural extension that is intentionally out of scope for
      this slice.
    * Gaussian scale -- estimated from the inlier (clustered) values by
      default. The ``std`` parameter lets a caller supply a known scale,
      honoring the paper's "known / independent covariance" model.
    * No sklearn / scipy -- DBSCAN and the truncated-normal CDF are
      implemented with numpy/math only, preserving the library's lightweight
      dependency footprint.
"""

import math
from typing import Any

import numpy as np

from dataframe_expectations.core.column_expectation import DataFrameColumnExpectation
from dataframe_expectations.core.types import ExpectationCategory, ExpectationSubcategory
from dataframe_expectations.core.utils import requires_params
from dataframe_expectations.registry import register_expectation

# Cap on candidate anomalies for which selective inference is computed. Beyond
# this many candidates the per-point line search becomes expensive; we then
# flag every noise point, which is the conservative outcome for a validation
# expectation (a column this noisy is genuinely off).
_MAX_ANOMALIES_FOR_SI = 50
# Floor for the estimated Gaussian scale to avoid division-by-zero on a column
# whose inliers are (near) constant.
_SIGMA_FLOOR = 1e-9
# Resolution of the parametric line search used to locate the selection set.
_SI_GRID = 201

_SQRT2 = math.sqrt(2.0)


def _normal_cdf(x: float) -> float:
    """Standard normal CDF, computed via :func:`math.erf` (scipy-free)."""
    return 0.5 * (1.0 + math.erf(x / _SQRT2))


def _truncated_normal_two_sided_pvalue(
    z_obs: float, sigma: float, intervals: list[tuple[float, float]]
) -> float:
    """Selective p-value ``P(|Z| >= |z_obs| | Z in union(intervals))``.

    ``Z`` is Gaussian with mean 0 and standard deviation ``sigma``;
    ``intervals`` is the (sorted) truncation set produced by the line search.
    The test is two-sided because an anomaly may lie in either tail.
    """

    def seg_mass(lo: float, hi: float) -> float:
        if not np.isfinite(lo):
            lo = -math.inf
        if not np.isfinite(hi):
            hi = math.inf
        if hi <= lo:
            return 0.0
        return _normal_cdf(hi / sigma) - _normal_cdf(lo / sigma)

    denom = sum(seg_mass(lo, hi) for lo, hi in intervals)
    if denom <= 0.0:
        return 0.0
    d = abs(z_obs)
    numer = 0.0
    for lo, hi in intervals:
        # Intersection of [lo, hi] with the two-sided tail (-inf, -d] U [d, inf).
        numer += seg_mass(lo, min(hi, -d))
        numer += seg_mass(max(lo, d), hi)
    return min(max(numer / denom, 0.0), 1.0)


def _dbscan_noise_mask_1d(values: np.ndarray, eps: float, min_samples: int) -> np.ndarray:
    """Boolean noise mask for univariate DBSCAN, aligned to ``values`` order.

    A point is noise iff it is neither a core point nor within ``eps`` of a
    core point (i.e. not density-reachable from any cluster). Implemented in
    sorted order so the neighbour lookups are O(n log n) and fully
    deterministic -- determinism is required so the same selection event is
    reproduced inside the selective-inference line search.
    """
    n = values.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(values, kind="stable")
    sv = values[order]
    left = np.searchsorted(sv, sv - eps, side="left")
    right = np.searchsorted(sv, sv + eps, side="right")
    core_counts = right - left  # points within eps, inclusive of self
    core_sorted = core_counts >= min_samples
    if core_sorted.any():
        core_vals = sv[core_sorted]
        pos = np.clip(np.searchsorted(core_vals, sv), 0, core_vals.shape[0] - 1)
        pos_lo = np.clip(pos - 1, 0, core_vals.shape[0] - 1)
        dist = np.minimum(np.abs(sv - core_vals[pos]), np.abs(sv - core_vals[pos_lo]))
        covered = dist <= eps
    else:
        covered = np.zeros(n, dtype=bool)
    in_cluster_sorted = core_sorted | covered
    noise_sorted = ~in_cluster_sorted
    noise = np.zeros(n, dtype=bool)
    noise[order] = noise_sorted
    return noise


def _default_eps(values: np.ndarray, min_samples: int) -> float | None:
    """Data-driven DBSCAN radius: mean gap between points ``min_samples`` apart.

    A simple k-distance-style heuristic that yields a small radius for dense
    columns (so genuine clusters form) without pulling in sklearn.
    """
    sv = np.sort(values)
    n = sv.shape[0]
    k = max(1, min(min_samples, n - 1))
    if n <= k:
        return None
    gaps = sv[k:] - sv[:-k]
    est = float(np.mean(gaps)) if gaps.size else None
    if est is None or not np.isfinite(est) or est <= 0:
        return None
    return est


def _selection_intervals(
    x: np.ndarray,
    j: int,
    inliers: np.ndarray,
    observed_noise: np.ndarray,
    t_obs: float,
    s_j: float,
    eps: float,
    min_samples: int,
    data_range: float,
    m: int,
) -> list[tuple[float, float]]:
    """Truncation set ``Z`` for candidate ``j`` via a parametric line search.

    Following the selective-inference construction, the data is reparameterised
    along the contrast direction as ``X(z) = c + d*z`` (``z`` is the value of
    the test statistic). ``Z`` is the set of ``z`` for which DBSCAN reproduces
    the *observed* anomaly set. The line is swept on a grid and the matching
    runs are returned as intervals; the dominant interval always contains the
    observed ``t_obs`` (the data itself reproduces its own selection).
    """
    dj = m / (m + 1.0)
    di = -1.0 / (m + 1.0)
    width = (data_range + 6.0 * eps) * (m + 1.0) / m + 12.0 * s_j
    if not np.isfinite(width) or width <= 0:
        width = 12.0 * s_j
    zs = np.linspace(t_obs - width, t_obs + width, num=_SI_GRID)
    deltas = zs - t_obs
    match = np.empty(zs.shape[0], dtype=bool)
    for k in range(zs.shape[0]):
        xz = x.copy()
        xz[j] = x[j] + dj * deltas[k]
        xz[inliers] = x[inliers] + di * deltas[k]
        match[k] = np.array_equal(_dbscan_noise_mask_1d(xz, eps, min_samples), observed_noise)
    intervals: list[tuple[float, float]] = []
    k = 0
    total = zs.shape[0]
    while k < total:
        if not match[k]:
            k += 1
            continue
        start = k
        while k < total and match[k]:
            k += 1
        intervals.append((float(zs[start]), float(zs[k - 1])))
    return intervals or [(t_obs, t_obs)]


def _anomaly_mask(
    values: np.ndarray,
    eps: float | None,
    min_samples: int,
    alpha: float,
    std: float | None,
) -> np.ndarray:
    """Boolean mask (aligned to ``values``) of statistically-significant anomalies."""
    full = np.asarray(values, dtype=float)
    flag = np.zeros(full.shape, dtype=bool)
    finite = np.where(~np.isnan(full))[0]
    if finite.shape[0] < 2:
        return flag
    x = full[finite]
    if eps is None:
        eps = _default_eps(x, min_samples)
        if eps is None:
            return flag
    noise = _dbscan_noise_mask_1d(x, eps, min_samples)
    anomalies = np.where(noise)[0]
    inliers = np.where(~noise)[0]
    if anomalies.shape[0] == 0 or inliers.shape[0] == 0:
        # No candidates, or no inliers to define a contrast against.
        return flag
    local_flag = np.zeros(x.shape[0], dtype=bool)
    if anomalies.shape[0] > _MAX_ANOMALIES_FOR_SI:
        # Too many candidates to run the per-point line search cheaply.
        local_flag[anomalies] = True
        flag[finite[local_flag]] = True
        return flag
    if std is not None:
        sigma = float(std)
    else:
        sigma = float(np.sqrt(np.var(x[inliers], ddof=0)))
    sigma = max(sigma, _SIGMA_FLOOR)
    m = inliers.shape[0]
    inlier_mean = float(np.mean(x[inliers]))
    data_range = float(np.max(x) - np.min(x))
    for j in anomalies:
        t_obs = float(x[j] - inlier_mean)
        s_j = sigma * math.sqrt(1.0 + 1.0 / m)
        intervals = _selection_intervals(
            x, int(j), inliers, noise, t_obs, s_j, eps, min_samples, data_range, m
        )
        if _truncated_normal_two_sided_pvalue(t_obs, s_j, intervals) <= alpha:
            local_flag[j] = True
    flag[finite[local_flag]] = True
    return flag


def _anomalous_rows_pandas(
    df: Any, column_name: str, eps: float | None, min_samples: int, alpha: float, std: float | None
) -> Any:
    values = df[column_name].to_numpy(dtype=float)
    return df[_anomaly_mask(values, eps, min_samples, alpha, std)]


def _anomalous_rows_polars(
    df: Any, column_name: str, eps: float | None, min_samples: int, alpha: float, std: float | None
) -> Any:
    import polars as pl

    values = np.asarray(df[column_name].to_numpy(), dtype=float)
    return df.filter(
        pl.Series("_si_clad_anomaly_flag", _anomaly_mask(values, eps, min_samples, alpha, std))
    )


def _anomalous_rows_pyspark(
    df: Any, column_name: str, eps: float | None, min_samples: int, alpha: float, std: float | None
) -> Any:
    # Selective inference is inherently eager (it needs the full column), so
    # the column is collected, scored in numpy, and the violating rows rebuilt
    # with the original schema.
    pdf = df.toPandas()
    values = pdf[column_name].to_numpy(dtype=float)
    violating = pdf.loc[_anomaly_mask(values, eps, min_samples, alpha, std)]
    if len(violating) == 0:
        return df.limit(0)
    return df.sparkSession.createDataFrame(violating, schema=df.schema)


@register_expectation(
    "ExpectationColumnValuesNotAnomalous",
    pydoc=(
        "Flag rows whose value in a numeric column is a statistically-significant "
        "anomaly. Candidate anomalies are DBSCAN noise points; each is kept only "
        "when its selective-inference p-value (SI-CLAD) is at most alpha, giving a "
        "per-anomaly false-detection probability of alpha instead of the uncontrolled "
        "rate of trusting raw DBSCAN noise."
    ),
    category=ExpectationCategory.COLUMN_EXPECTATIONS,
    subcategory=ExpectationSubcategory.NUMERICAL,
    params_doc={
        "column_name": "Numeric column to scan for anomalous values.",
        "eps": "DBSCAN neighbourhood radius. If None, derived from the column.",
        "min_samples": "DBSCAN core-point threshold (including the point itself).",
        "alpha": "Significance level; a candidate is flagged when its p-value <= alpha.",
        "std": "Known Gaussian scale of inliers. If None, estimated from clustered values.",
    },
)
@requires_params("column_name", types={"column_name": str})
def create_expectation_column_values_not_anomalous(
    column_name: str,
    eps: float | None = None,
    min_samples: int = 5,
    alpha: float = 0.05,
    std: float | None = None,
    tags: list[str] | None = None,
) -> DataFrameColumnExpectation:
    return DataFrameColumnExpectation(
        expectation_name="ExpectationColumnValuesNotAnomalous",
        column_name=column_name,
        fn_violations_pandas=lambda df: _anomalous_rows_pandas(
            df, column_name, eps, min_samples, alpha, std
        ),
        fn_violations_pyspark=lambda df: _anomalous_rows_pyspark(
            df, column_name, eps, min_samples, alpha, std
        ),
        fn_violations_polars=lambda df: _anomalous_rows_polars(
            df, column_name, eps, min_samples, alpha, std
        ),
        description=(
            f"'{column_name}' values are not statistically-significant anomalies "
            f"(SI-CLAD, alpha={alpha})"
        ),
        error_message=f"'{column_name}' has statistically-significant anomalous values",
        tags=tags,
    )
