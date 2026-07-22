"""Cross-column association rule mining and validation.

Adapted from "Neurosymbolic Association Rule Mining from Tabular Data" (Aerial+,
arXiv:2504.19354). The paper mines logical cross-column rules from tabular data
and uses them as data-quality constraints. This module keeps that core
mechanism — association rule mining scored by *support* and *confidence*, with
the mined rules exposed as a registered expectation — and substitutes the
paper's learned under-complete neurosymbolic autoencoder (used to compress
item representations and prune rule explosion) with a parameter-free,
Apriori-style frequency/confidence pruner. Mining is a fit-time operation over
a pandas reference DataFrame; validation runs on all supported backends.
"""

import itertools
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import pandas as pd
from pandas import DataFrame as PandasDataFrame

from dataframe_expectations.core.aggregation_expectation import (
    DataFrameAggregationExpectation,
)
from dataframe_expectations.core.polars_utils import PolarsDataFrame, get_polars_functions
from dataframe_expectations.core.pyspark_utils import (
    PySparkDataFrame,
    get_pyspark_functions,
)
from dataframe_expectations.core.types import (
    DataFrameLike,
    DataFrameType,
    ExpectationCategory,
    ExpectationSubcategory,
)
from dataframe_expectations.registry import register_expectation
from dataframe_expectations.result_message import (
    DataFrameExpectationFailureMessage,
    DataFrameExpectationResultMessage,
    DataFrameExpectationSuccessMessage,
)

# Lazy proxies so importing this module never requires pyspark/polars at load time.
F_PYSPARK = get_pyspark_functions()
F_POLARS = get_polars_functions()


def _fmt_value(value: Any) -> str:
    """Render a rule term value for human-readable rule strings."""
    return f'"{value}"' if isinstance(value, str) else str(value)


@dataclass(frozen=True)
class AssociationRule:
    """A mined cross-column implication rule: antecedent => consequent.

    :ivar antecedent: Column/value pairs that must jointly hold (the "if" side).
    :ivar consequent: Column/value pairs expected to hold whenever the
        antecedent holds (the "then" side).
    :ivar support: Fraction of rows where the full rule (antecedent AND
        consequent) holds — i.e. P(antecedent, consequent).
    :ivar confidence: Conditional probability P(consequent | antecedent).
    """

    antecedent: Dict[str, Any]
    consequent: Dict[str, Any]
    support: float
    confidence: float

    def __str__(self) -> str:
        antecedent = " AND ".join(
            f"{col}={_fmt_value(val)}" for col, val in self.antecedent.items()
        )
        consequent = " AND ".join(
            f"{col}={_fmt_value(val)}" for col, val in self.consequent.items()
        )
        return (
            f"IF {antecedent} THEN {consequent} "
            f"(support={self.support:.3f}, confidence={self.confidence:.3f})"
        )


def _freeze(mapping: Dict[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    """Hashable representation of a column/value mapping for deduplication."""
    return tuple(sorted(mapping.items(), key=lambda item: item[0]))


def _pandas_match_mask(data_frame: PandasDataFrame, items: Sequence[Tuple[str, Any]]) -> pd.Series:
    """Boolean mask of rows where every (column, value) pair in ``items`` holds."""
    if not items:
        return pd.Series([True] * len(data_frame), index=data_frame.index)
    mask: Optional[pd.Series] = None
    for column, value in items:
        column_mask = (data_frame[column] == value).fillna(False)
        mask = column_mask if mask is None else (mask & column_mask)
    return cast(pd.Series, mask)


def mine_association_rules(
    data_frame: PandasDataFrame,
    columns: Optional[List[str]] = None,
    min_support: float = 0.05,
    min_confidence: float = 0.8,
    max_antecedent_size: int = 2,
    max_cardinality: int = 50,
    max_rules: Optional[int] = None,
) -> List[AssociationRule]:
    """Mine cross-column association rules from a reference pandas DataFrame.

    This is the parameter-free substitute for Aerial+'s learned rule miner:
    frequent single items are enumerated, bounded itemsets are assembled
    Apriori-style (an itemset is explored only when every member is frequent),
    and each frequent antecedent is paired with frequent single-item consequents
    whose conditional probability clears ``min_confidence``. Rule explosion is
    bounded by ``min_support``, ``max_antecedent_size`` and ``max_cardinality``
    rather than by the paper's under-complete autoencoder.

    :param data_frame: Reference pandas DataFrame to mine rules from.
    :param columns: Columns to consider. Defaults to every column. Columns with
        more distinct values than ``max_cardinality`` (e.g. identifiers) are
        skipped to avoid explosion.
    :param min_support: Minimum fraction of rows an item/itemset must cover to
        be considered frequent.
    :param min_confidence: Minimum P(consequent | antecedent) to emit a rule.
    :param max_antecedent_size: Largest antecedent itemset size to explore.
    :param max_cardinality: Skip columns whose distinct-value count exceeds this.
    :param max_rules: If set, keep only this many highest-confidence rules.
    :return: Mined rules sorted by confidence (desc) then support (desc).
    """
    if min_support <= 0 or min_support > 1:
        raise ValueError(f"min_support must be in (0, 1], got {min_support}")
    if min_confidence <= 0 or min_confidence > 1:
        raise ValueError(f"min_confidence must be in (0, 1], got {min_confidence}")

    selected_columns = columns if columns is not None else list(data_frame.columns)
    total_rows = len(data_frame)
    if total_rows == 0:
        return []

    frequent_items: List[Tuple[str, Any]] = []
    for column in selected_columns:
        value_counts = data_frame[column].value_counts(dropna=True)
        if len(value_counts) > max_cardinality:
            continue
        for value, count in value_counts.items():
            if count / total_rows >= min_support:
                frequent_items.append((column, value))

    if not frequent_items:
        return []

    rules: List[AssociationRule] = []
    for size in range(1, max_antecedent_size + 1):
        for antecedent in itertools.combinations(frequent_items, size):
            antecedent_columns = {column for column, _ in antecedent}
            # Items in one itemset must come from distinct columns.
            if len(antecedent_columns) != size:
                continue
            antecedent_mask = _pandas_match_mask(data_frame, antecedent)
            antecedent_count = int(antecedent_mask.sum())
            if antecedent_count == 0 or antecedent_count / total_rows < min_support:
                continue
            for consequent_column, consequent_value in frequent_items:
                if consequent_column in antecedent_columns:
                    continue
                joint_count = int(
                    (antecedent_mask & (data_frame[consequent_column] == consequent_value))
                    .fillna(False)
                    .sum()
                )
                if joint_count == 0:
                    continue
                confidence = joint_count / antecedent_count
                if confidence >= min_confidence:
                    rules.append(
                        AssociationRule(
                            antecedent={column: value for column, value in antecedent},
                            consequent={consequent_column: consequent_value},
                            support=joint_count / total_rows,
                            confidence=confidence,
                        )
                    )

    seen: set = set()
    unique_rules: List[AssociationRule] = []
    for rule in rules:
        key = (_freeze(rule.antecedent), _freeze(rule.consequent))
        if key in seen:
            continue
        seen.add(key)
        unique_rules.append(rule)

    unique_rules.sort(key=lambda rule: (rule.confidence, rule.support), reverse=True)
    if max_rules is not None:
        unique_rules = unique_rules[:max_rules]
    return unique_rules


def _match_column(items: Sequence[Tuple[str, Any]], functions: Any) -> Any:
    """Build a combined equality condition (pyspark Column / polars Expr)."""
    if not items:
        return functions.lit(True)
    condition = None
    for column, value in items:
        equality = functions.col(column) == value
        condition = equality if condition is None else (condition & equality)
    return condition


def _rule_violation_condition(rule: AssociationRule, functions: Any) -> Any:
    """Condition matching rows that break ``rule`` (antecedent holds, consequent does not)."""
    antecedent = _match_column(tuple(rule.antecedent.items()), functions)
    consequent = _match_column(tuple(rule.consequent.items()), functions)
    return antecedent & ~consequent


class ExpectationCrossColumnRules(DataFrameAggregationExpectation):
    """Expectation enforcing a set of cross-column association rules per row.

    Each row must satisfy every rule: whenever a row matches a rule's antecedent
    it must also match that rule's consequent. Rows matching an antecedent but
    violating the consequent are returned as violations. Rules are typically
    produced by :func:`mine_association_rules` but may be hand-authored.
    """

    def __init__(
        self, rules: Optional[List[AssociationRule]] = None, tags: Optional[List[str]] = None
    ) -> None:
        """Initialize the cross-column rule expectation.

        :param rules: Association rules to enforce as per-row constraints. When
            omitted or empty the expectation trivially succeeds (no constraints).
        :param tags: Optional tags as list of strings in "key:value" format.
        """
        self.rules: List[AssociationRule] = list(rules) if rules else []
        for rule in self.rules:
            if not isinstance(rule, AssociationRule):
                raise TypeError(
                    "rules must be a list of AssociationRule, "
                    f"got element of type {type(rule).__name__}"
                )
        referenced_columns = sorted(
            {column for rule in self.rules for column in (*rule.antecedent, *rule.consequent)}
        )
        super().__init__(
            expectation_name="ExpectationCrossColumnRules",
            column_names=referenced_columns,
            description=f"DataFrame obeys {len(self.rules)} cross-column association rule(s)",
            tags=tags,
        )

    def aggregate_and_validate_pandas(
        self, data_frame: DataFrameLike, **kwargs
    ) -> DataFrameExpectationResultMessage:
        """Validate a pandas DataFrame against the rule set."""
        pandas_df = cast(PandasDataFrame, data_frame)
        if not self.rules:
            return DataFrameExpectationSuccessMessage(expectation_name=self.get_expectation_name())
        combined: Optional[pd.Series] = None
        for rule in self.rules:
            antecedent = _pandas_match_mask(pandas_df, tuple(rule.antecedent.items()))
            consequent = _pandas_match_mask(pandas_df, tuple(rule.consequent.items()))
            violation = antecedent & ~consequent
            combined = violation if combined is None else (combined | violation)
        violations = pandas_df[cast(pd.Series, combined)]
        return self._build_result(
            violations=violations,
            row_count=len(violations),
            data_frame_type=DataFrameType.PANDAS,
        )

    def aggregate_and_validate_pyspark(
        self, data_frame: DataFrameLike, **kwargs
    ) -> DataFrameExpectationResultMessage:
        """Validate a PySpark DataFrame against the rule set."""
        pyspark_df = cast(PySparkDataFrame, data_frame)
        if not self.rules:
            return DataFrameExpectationSuccessMessage(expectation_name=self.get_expectation_name())
        combined = None
        for rule in self.rules:
            condition = _rule_violation_condition(rule, F_PYSPARK)
            combined = condition if combined is None else (combined | condition)
        violations = pyspark_df.filter(combined)
        return self._build_result(
            violations=violations,
            row_count=violations.count(),
            data_frame_type=DataFrameType.PYSPARK,
        )

    def aggregate_and_validate_polars(
        self, data_frame: DataFrameLike, **kwargs
    ) -> DataFrameExpectationResultMessage:
        """Validate a Polars DataFrame against the rule set."""
        polars_df = cast(PolarsDataFrame, data_frame)
        if not self.rules:
            return DataFrameExpectationSuccessMessage(expectation_name=self.get_expectation_name())
        combined = None
        for rule in self.rules:
            expression = _rule_violation_condition(rule, F_POLARS)
            combined = expression if combined is None else (combined | expression)
        violations = polars_df.filter(combined)
        return self._build_result(
            violations=violations,
            row_count=violations.height,
            data_frame_type=DataFrameType.POLARS,
        )

    def _build_result(
        self,
        violations: DataFrameLike,
        row_count: int,
        data_frame_type: DataFrameType,
    ) -> DataFrameExpectationResultMessage:
        if row_count == 0:
            return DataFrameExpectationSuccessMessage(expectation_name=self.get_expectation_name())
        return DataFrameExpectationFailureMessage(
            expectation_str=str(self),
            data_frame_type=data_frame_type,
            violations_data_frame=violations,
            message=(
                f"Found {row_count} row(s) violating "
                f"{len(self.rules)} cross-column association rule(s)."
            ),
        )


@register_expectation(
    "ExpectationCrossColumnRules",
    pydoc=(
        "Validate that every row obeys a set of cross-column association rules, "
        "typically mined from reference data via mine_association_rules."
    ),
    category=ExpectationCategory.COLUMN_AGGREGATION_EXPECTATIONS,
    subcategory=ExpectationSubcategory.ANY_VALUE,
    params_doc={
        "rules": "List of AssociationRule objects to enforce as cross-column constraints.",
    },
)
def create_expectation_cross_column_rules(
    rules: Optional[List[AssociationRule]] = None, tags: Optional[List[str]] = None
) -> ExpectationCrossColumnRules:
    """Create an ExpectationCrossColumnRules instance.

    :param rules: Association rules to enforce as per-row constraints.
    :param tags: Optional tags as list of strings in "key:value" format.
    :return: A configured expectation instance.
    """
    return ExpectationCrossColumnRules(rules=rules, tags=tags)
