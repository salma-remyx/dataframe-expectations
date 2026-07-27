"""Integration tests for the SI-CLAD anomaly-detection column expectation.

These tests drive the expectation entirely through the repo's existing public
API (the registry + the suite) rather than the new module's internals, which
is what proves the new capability is wired into the library.
"""

import pytest

from dataframe_expectations.registry import DataFrameExpectationRegistry
from dataframe_expectations.result_message import (
    DataFrameExpectationFailureMessage,
    DataFrameExpectationSuccessMessage,
)
from dataframe_expectations.suite import (
    DataFrameExpectationsSuite,
    DataFrameExpectationsSuiteFailure,
)

# A dense cluster of typical values plus three clear outliers.
CLUSTER = [48.0, 49.0, 50.0, 51.0, 52.0, 48.5, 49.5, 50.5, 51.5, 47.5, 52.5, 50.0]
OUTLIERS = [200.0, -100.0, 500.0]
EPS = 1.5
MIN_SAMPLES = 4

EXPECTATION_NAME = "ExpectationColumnValuesNotAnomalous"


def _make_column(make_df, values):
    return make_df({"col1": (values, "double")})


def test_expectation_is_registered():
    """The new expectation is auto-discovered and exposed on the suite."""
    assert EXPECTATION_NAME in DataFrameExpectationRegistry.list_expectations()
    assert (
        "expect_column_values_not_anomalous"
        in DataFrameExpectationRegistry.get_suite_method_mapping()
    )


def test_clear_anomalies_are_flagged(dataframe_factory):
    """Rows with extreme outlier values are reported as violations."""
    df_lib, make_df = dataframe_factory
    data_frame = _make_column(make_df, CLUSTER + OUTLIERS)

    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name=EXPECTATION_NAME,
        column_name="col1",
        eps=EPS,
        min_samples=MIN_SAMPLES,
        alpha=0.05,
    )
    result = expectation.validate(data_frame=data_frame)

    assert isinstance(result, DataFrameExpectationFailureMessage)
    assert "Found 3 row(s)" in str(result)

    # On pandas we can also confirm exactly *which* rows were flagged.
    if df_lib.value == "Pandas":
        violations = result.violations_data_frame
        flagged = set(violations["col1"].tolist())
        assert flagged == set(OUTLIERS)


def test_clean_column_passes(dataframe_factory):
    """A column with only the typical cluster has no significant anomalies."""
    _df_lib, make_df = dataframe_factory
    data_frame = _make_column(make_df, CLUSTER)

    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name=EXPECTATION_NAME,
        column_name="col1",
        eps=EPS,
        min_samples=MIN_SAMPLES,
        alpha=0.05,
    )
    result = expectation.validate(data_frame=data_frame)
    assert isinstance(result, DataFrameExpectationSuccessMessage)


def test_significance_gates_detection(dataframe_factory):
    """The selective p-value -- not raw DBSCAN -- drives the verdict.

    Two tight clusters plus an isolated point between them: DBSCAN labels that
    middle point as noise (a candidate anomaly) regardless of the noise scale.
    Whether it is *flagged* depends on its selective-inference p-value, which is
    a function of the assumed Gaussian noise scale (``std``). With a large scale
    the point is indistinguishable from noise (not flagged); with a tiny scale
    it is highly significant (flagged). This is the core contribution of
    SI-CLAD: statistical significance gates the raw clustering output.
    """
    _df_lib, make_df = dataframe_factory
    # 5-point cluster near 0, 5-point cluster near 10, one isolated point at 5.
    data = [0.0, 0.1, 0.2, 0.3, 0.4, 10.0, 10.1, 10.2, 10.3, 10.4, 5.0]
    data_frame = make_df({"col1": (data, "double")})

    common = {
        "expectation_name": EXPECTATION_NAME,
        "column_name": "col1",
        "eps": 0.3,
        "min_samples": 4,
        "alpha": 0.05,
    }

    not_significant = DataFrameExpectationRegistry.get_expectation(std=1.0, **common)
    assert isinstance(
        not_significant.validate(data_frame=data_frame), DataFrameExpectationSuccessMessage
    )

    significant = DataFrameExpectationRegistry.get_expectation(std=0.01, **common)
    result = significant.validate(data_frame=data_frame)
    assert isinstance(result, DataFrameExpectationFailureMessage)
    assert "Found 1 row(s)" in str(result)


def test_suite_runs_anomaly_expectation(dataframe_factory):
    """The expectation is usable through the suite builder/run path."""
    _df_lib, make_df = dataframe_factory

    clean_frame = _make_column(make_df, CLUSTER)
    dirty_frame = _make_column(make_df, CLUSTER + OUTLIERS)

    clean_suite = (
        DataFrameExpectationsSuite()
        .expect_column_values_not_anomalous(
            column_name="col1", eps=EPS, min_samples=MIN_SAMPLES, alpha=0.05
        )
        .build()
    )
    assert clean_suite.run(data_frame=clean_frame).success

    dirty_suite = (
        DataFrameExpectationsSuite()
        .expect_column_values_not_anomalous(
            column_name="col1", eps=EPS, min_samples=MIN_SAMPLES, alpha=0.05
        )
        .build()
    )
    with pytest.raises(DataFrameExpectationsSuiteFailure):
        dirty_suite.run(data_frame=dirty_frame)


def test_missing_column_reports_error(dataframe_factory):
    """A missing column yields a clear failure message via the base contract."""
    _df_lib, make_df = dataframe_factory
    data_frame = make_df({"other": ([1.0, 2.0, 3.0], "double")})

    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name=EXPECTATION_NAME,
        column_name="col1",
        eps=EPS,
        min_samples=MIN_SAMPLES,
        alpha=0.05,
    )
    result = expectation.validate(data_frame=data_frame)
    assert isinstance(result, DataFrameExpectationFailureMessage)
    assert "Column 'col1' does not exist" in str(result)
