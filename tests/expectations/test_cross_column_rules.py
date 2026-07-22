"""Integration tests for the cross-column association rule expectation.

These tests go through the existing public surface — the registry's
auto-discovery and the suite's dynamic ``expect_*`` forward path — rather than
the new module in isolation. They prove the mined rules are wired into the
existing validation pipeline as a registered expectation.
"""

import pandas as pd
import pytest

from dataframe_expectations.core.suite_result import SuiteExecutionResult
from dataframe_expectations.core.types import DataFrameType
from dataframe_expectations.expectations.cross_column_rules import (
    AssociationRule,
    mine_association_rules,
)
from dataframe_expectations.registry import DataFrameExpectationRegistry
from dataframe_expectations.result_message import (
    DataFrameExpectationSuccessMessage,
)
from dataframe_expectations.suite import (
    DataFrameExpectationsSuite,
    DataFrameExpectationsSuiteFailure,
)

EXPECTATION_NAME = "ExpectationCrossColumnRules"


@pytest.fixture()
def reference_dataframe() -> pd.DataFrame:
    """Reference data where country deterministically implies currency."""
    return pd.DataFrame(
        {
            "country": ["US", "US", "US", "US", "DE", "DE", "DE"],
            "currency": ["USD", "USD", "USD", "USD", "EUR", "EUR", "EUR"],
        }
    )


@pytest.fixture()
def mined_rules(reference_dataframe):
    return mine_association_rules(reference_dataframe, min_support=0.3, min_confidence=0.8)


def test_expectation_is_auto_discovered_by_registry():
    """The new module is imported and registered by the existing registry loader."""
    assert EXPECTATION_NAME in DataFrameExpectationRegistry.list_expectations()
    mapping = DataFrameExpectationRegistry.get_suite_method_mapping()
    assert mapping["expect_cross_column_rules"] == EXPECTATION_NAME


def test_mine_association_rules_keeps_high_confidence_rules(mined_rules):
    """Mining recovers deterministic implications and drops weak ones."""
    antecedents = [rule.antecedent for rule in mined_rules]
    assert {"country": "US"} in antecedents
    assert {"country": "DE"} in antecedents

    us_rule = next(rule for rule in mined_rules if rule.antecedent == {"country": "US"})
    assert us_rule.consequent == {"currency": "USD"}
    assert us_rule.confidence == pytest.approx(1.0)
    assert us_rule.support == pytest.approx(4 / 7)

    # With EUR present only on DE rows, currency=EUR => country=DE is also deterministic.
    eur_rule = next(rule for rule in mined_rules if rule.antecedent == {"currency": "EUR"})
    assert eur_rule.consequent == {"country": "DE"}


def test_mine_association_rules_rejects_invalid_thresholds(reference_dataframe):
    with pytest.raises(ValueError):
        mine_association_rules(reference_dataframe, min_support=0.0)
    with pytest.raises(ValueError):
        mine_association_rules(reference_dataframe, min_confidence=1.5)


def test_expectation_via_registry_passes_and_fails(mined_rules):
    rules = [rule for rule in mined_rules if rule.antecedent == {"country": "US"}]
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name=EXPECTATION_NAME, rules=rules
    )

    conforming = pd.DataFrame({"country": ["US", "DE"], "currency": ["USD", "EUR"]})
    violating = pd.DataFrame({"country": ["US", "DE"], "currency": ["EUR", "EUR"]})

    success = expectation.validate(data_frame=conforming)
    assert str(success) == str(
        DataFrameExpectationSuccessMessage(expectation_name=EXPECTATION_NAME)
    )

    failure = expectation.validate(data_frame=violating)
    assert "violating" in str(failure)
    violations = failure.get_violations_data_frame()
    assert len(violations) == 1


def test_expectation_via_suite_forward_path(mined_rules):
    rules = [rule for rule in mined_rules if rule.antecedent == {"country": "US"}]
    conforming = pd.DataFrame({"country": ["US", "DE"], "currency": ["USD", "EUR"]})
    violating = pd.DataFrame({"country": ["US", "DE"], "currency": ["EUR", "EUR"]})

    suite = DataFrameExpectationsSuite().expect_cross_column_rules(rules=rules)

    result = suite.build().run(data_frame=conforming)
    assert isinstance(result, SuiteExecutionResult)
    assert result.success
    assert result.total_passed == 1
    assert result.total_failed == 0

    with pytest.raises(DataFrameExpectationsSuiteFailure):
        suite.build().run(data_frame=violating)


def test_empty_rule_set_always_passes():
    """No rules means no constraints, so validation always succeeds."""
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name=EXPECTATION_NAME, rules=[]
    )
    df = pd.DataFrame({"a": [1, 2, 3]})
    assert "succeeded" in str(expectation.validate(data_frame=df))


def test_factory_rejects_non_rule_items():
    """When rules are provided, every element must be an AssociationRule."""
    with pytest.raises(TypeError):
        DataFrameExpectationRegistry.get_expectation(
            expectation_name=EXPECTATION_NAME, rules=["not", "rules"]
        )


@pytest.mark.parametrize(
    "df_data, expect_success, expected_violations",
    [
        # Conforming: every antecedent row satisfies its consequent.
        (
            {"country": (["US", "DE"], "string"), "currency": (["USD", "EUR"], "string")},
            True,
            0,
        ),
        # Violating: US row claims EUR currency.
        (
            {"country": (["US", "DE"], "string"), "currency": (["EUR", "EUR"], "string")},
            False,
            1,
        ),
    ],
    ids=["conforming", "violating"],
)
def test_validation_across_backends(
    dataframe_factory, mined_rules, df_data, expect_success, expected_violations
):
    """The mined rules validate against every supported DataFrame backend."""
    df_lib, make_df = dataframe_factory
    rules = [rule for rule in mined_rules if rule.antecedent == {"country": "US"}]
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name=EXPECTATION_NAME, rules=rules
    )

    data_frame = make_df(df_data)
    result = expectation.validate(data_frame=data_frame)

    if expect_success:
        assert str(result) == str(
            DataFrameExpectationSuccessMessage(expectation_name=EXPECTATION_NAME)
        ), f"[{df_lib.value}] expected success but got: {result}"
    else:
        assert "violating" in str(result), f"[{df_lib.value}] expected failure but got: {result}"
        violations = result.get_violations_data_frame()
        assert violations is not None
        row_count = len(violations) if df_lib == DataFrameType.PANDAS else violations.count()
        assert row_count == expected_violations


def test_hand_authored_rule_validates():
    """Rules need not be mined — a hand-built implication is enforced too."""
    rule = AssociationRule(
        antecedent={"status": "active"},
        consequent={"enabled": "true"},
        support=1.0,
        confidence=1.0,
    )
    expectation = DataFrameExpectationRegistry.get_expectation(
        expectation_name=EXPECTATION_NAME, rules=[rule]
    )

    conforming = pd.DataFrame({"status": ["active", "inactive"], "enabled": ["true", "false"]})
    violating = pd.DataFrame({"status": ["active", "inactive"], "enabled": ["false", "false"]})

    assert "succeeded" in str(expectation.validate(data_frame=conforming))
    failure = expectation.validate(data_frame=violating)
    assert "violating" in str(failure)
    assert len(failure.get_violations_data_frame()) == 1
