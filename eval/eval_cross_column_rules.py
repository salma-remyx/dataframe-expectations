"""Evaluation: cross-column association-rule mining (Apriori) recall + determinism.

Builds a synthetic 600-row reference DataFrame with 5 planted deterministic
single-column implications (country -> currency x3, plan -> tier x2) plus a
high-cardinality noise column that the miner must skip. Measures:

- cross_column_rule_recall: fraction of the 5 planted implications recovered
  by dataframe_expectations.expectations.cross_column_rules.mine_association_rules
  with method="apriori".
- cross_column_mining_determinism_rate: whether two mining calls on identical
  input produce an identical rule set (Apriori has no RNG / training loop).

On baseline (changed module unimportable), mining falls back to an empty list
on both calls, so recall degrades to 0.0 while determinism trivially stays 1.0
(both empty results match) — the guardrail can never fail on baseline.
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

# Put repo root on sys.path so the real package is importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dataframe_expectations.expectations.cross_column_rules import (
        mine_association_rules as _mine_association_rules,
    )

    HAVE_FEATURE = True
except Exception:
    HAVE_FEATURE = False

    def _mine_association_rules(*args, **kwargs):
        return []


def build_reference_frame(rows: int = 600, seed: int = 0) -> pd.DataFrame:
    """600-row frame with 5 planted deterministic implications + noise column."""
    rng = np.random.default_rng(seed)
    country_to_currency = {"US": "USD", "DE": "EUR", "JP": "JPY"}
    plan_to_tier = {"basic": "bronze", "premium": "gold"}

    country_col = rng.choice(list(country_to_currency.keys()), size=rows)
    plan_col = rng.choice(list(plan_to_tier.keys()), size=rows)
    currency_col = [country_to_currency[c] for c in country_col]
    tier_col = [plan_to_tier[p] for p in plan_col]
    # High-cardinality identifier column the miner must skip (max_cardinality=50).
    noise_col = rng.integers(0, 1000, size=rows)

    return pd.DataFrame(
        {
            "country": country_col,
            "currency": currency_col,
            "plan": plan_col,
            "tier": tier_col,
            "noise_id": noise_col,
        }
    )


def mine(data_frame: pd.DataFrame):
    """Call the real mine_association_rules with method='apriori' explicitly."""
    return _mine_association_rules(
        data_frame,
        min_support=0.05,
        min_confidence=0.9,
        max_antecedent_size=2,
        max_cardinality=50,
        method="apriori",
    )


def rule_key(rule):
    """Hashable (antecedent, consequent) anchor for a mined rule."""
    antecedent = rule.antecedent if hasattr(rule, "antecedent") else rule[0]
    consequent = rule.consequent if hasattr(rule, "consequent") else rule[1]
    return (frozenset(antecedent.items()), frozenset(consequent.items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", default=None)
    parser.add_argument("--ref", default=None)
    parser.add_argument("--seed", default=None)
    parser.parse_args()

    data_frame = build_reference_frame()

    planted = [
        ({"country": "US"}, {"currency": "USD"}),
        ({"country": "DE"}, {"currency": "EUR"}),
        ({"country": "JP"}, {"currency": "JPY"}),
        ({"plan": "basic"}, {"tier": "bronze"}),
        ({"plan": "premium"}, {"tier": "gold"}),
    ]
    planted_keys = {
        (frozenset(a.items()), frozenset(c.items())) for a, c in planted
    }

    try:
        rules_run1 = mine(data_frame)
    except Exception:
        rules_run1 = []
    try:
        rules_run2 = mine(data_frame)
    except Exception:
        rules_run2 = []

    mined_keys = set()
    for rule in rules_run1:
        try:
            mined_keys.add(rule_key(rule))
        except Exception:
            continue

    recovered = sum(1 for key in planted_keys if key in mined_keys)
    recall = recovered / len(planted_keys)

    try:
        keys1 = sorted(str(rule_key(rule)) for rule in rules_run1)
        keys2 = sorted(str(rule_key(rule)) for rule in rules_run2)
        determinism_rate = 1.0 if keys1 == keys2 else 0.0
    except Exception:
        determinism_rate = 0.0

    if not HAVE_FEATURE:
        recall = 0.0
        determinism_rate = 1.0  # both calls trivially return [] == []

    print(
        json.dumps(
            {
                "cross_column_rule_recall": recall,
                "cross_column_mining_determinism_rate": determinism_rate,
            }
        )
    )


if __name__ == "__main__":
    main()