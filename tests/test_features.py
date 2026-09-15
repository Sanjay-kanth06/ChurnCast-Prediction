"""Feature engineering correctness, with emphasis on the leakage guard."""
from __future__ import annotations

import pandas as pd

from src.features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES
from src.features.features import (
    FORBIDDEN_SOURCE_COLUMNS, _log_features, _transaction_features,
)


def test_all_declared_features_present(feature_table):
    missing = [c for c in FEATURE_COLUMNS if c not in feature_table.columns]
    assert not missing, f"missing features: {missing}"


def test_label_present_and_binary(feature_table):
    assert "is_churn" in feature_table.columns
    assert set(feature_table["is_churn"].unique()).issubset({0, 1})


def test_one_row_per_labelled_subscriber(feature_table, raw_tables):
    assert len(feature_table) == len(raw_tables["train"])
    assert feature_table.index.is_unique


def test_no_null_explosion(feature_table):
    """No feature may be mostly missing. `age` is the known exception: the
    generator emits 0 for unknown ages, mirroring KKBOX, and those become null."""
    nulls = feature_table[FEATURE_COLUMNS].isna().mean()
    bad = nulls[(nulls > 0.50) & (nulls.index != "age")]
    assert bad.empty, f"features over 50% null: {dict(bad)}"


def test_no_constant_features(feature_table):
    constant = [c for c in NUMERIC_FEATURES if feature_table[c].nunique(dropna=True) <= 1]
    assert not constant, f"constant features carry no information: {constant}"


def test_ratio_features_are_bounded(feature_table):
    for col in ["skip_ratio", "completion_ratio"]:
        s = feature_table[col].dropna()
        assert s.between(0, 1).all(), f"{col} outside [0, 1]"


def test_counts_are_non_negative(feature_table):
    for col in ["txns_last_30d", "txns_last_90d", "active_days_last_30d",
                "active_days_last_90d", "total_secs_last_30d", "prior_cancellations",
                "tenure_days", "recency_days"]:
        s = feature_table[col].dropna()
        assert (s >= 0).all(), f"{col} contains negative values"


def test_window_features_are_ordered(feature_table):
    """A 30-day window can never exceed its enclosing 90-day window."""
    assert (feature_table["active_days_last_30d"]
            <= feature_table["active_days_last_90d"]).all()
    assert (feature_table["total_secs_last_30d"]
            <= feature_table["total_secs_last_90d"] + 1e-6).all()


def test_active_days_within_window_length(feature_table):
    assert feature_table["active_days_last_30d"].max() <= 31
    assert feature_table["active_days_last_90d"].max() <= 91


def test_forbidden_columns_are_not_features():
    """membership_expire_date defines the label; it must never be a feature."""
    for col in FORBIDDEN_SOURCE_COLUMNS:
        assert col not in FEATURE_COLUMNS


def test_categorical_features_are_strings(feature_table):
    for col in CATEGORICAL_FEATURES:
        non_null = feature_table[col].dropna()
        assert non_null.map(lambda v: isinstance(v, str)).all(), f"{col} is not string-typed"


# --------------------------------------------------------------- leakage guard

def _fixture_frames():
    cutoff = pd.Timestamp("2024-07-15")
    cohorts = pd.DataFrame({"msno": ["A"], "observation_cutoff": [cutoff]})
    tx = pd.DataFrame({
        "msno": ["A", "A"],
        "payment_method_id": [41, 41],
        "payment_plan_days": [30, 30],
        "plan_list_price": [149, 149],
        "actual_amount_paid": [149, 149],
        "is_auto_renew": [1, 1],
        "transaction_date": [pd.Timestamp("2024-07-01"), pd.Timestamp("2024-06-01")],
        "membership_expire_date": [cutoff, pd.Timestamp("2024-07-01")],
        "is_cancel": [0, 0],
    })
    logs = pd.DataFrame({
        "msno": ["A"] * 3,
        "date": [pd.Timestamp("2024-07-10"), pd.Timestamp("2024-07-05"),
                 pd.Timestamp("2024-06-20")],
        "num_25": [1, 2, 1], "num_50": [1, 1, 1], "num_75": [1, 1, 1],
        "num_985": [1, 1, 1], "num_100": [10, 12, 9], "num_unq": [8, 9, 7],
        "total_secs": [2000.0, 2400.0, 1800.0],
    })
    return cohorts, tx, logs, cutoff


def test_post_cutoff_transactions_are_ignored():
    """Adding a transaction AFTER the cutoff must not change any feature.
    This is the guard that prevents the renewal from leaking into training."""
    cohorts, tx, logs, cutoff = _fixture_frames()
    before = _transaction_features(tx, cohorts)

    leaked = pd.concat([tx, pd.DataFrame([{
        "msno": "A", "payment_method_id": 99, "payment_plan_days": 410,
        "plan_list_price": 9999, "actual_amount_paid": 9999, "is_auto_renew": 0,
        "transaction_date": cutoff + pd.Timedelta(days=5),
        "membership_expire_date": cutoff + pd.Timedelta(days=415), "is_cancel": 1,
    }])], ignore_index=True)
    after = _transaction_features(leaked, cohorts)

    pd.testing.assert_frame_equal(before, after)


def test_post_cutoff_logs_are_ignored():
    cohorts, tx, logs, cutoff = _fixture_frames()
    before = _log_features(logs, cohorts)

    leaked = pd.concat([logs, pd.DataFrame([{
        "msno": "A", "date": cutoff + pd.Timedelta(days=3),
        "num_25": 500, "num_50": 500, "num_75": 500, "num_985": 500,
        "num_100": 500, "num_unq": 500, "total_secs": 999999.0,
    }])], ignore_index=True)
    after = _log_features(leaked, cohorts)

    pd.testing.assert_frame_equal(before, after)


def test_recency_is_measured_from_the_cutoff():
    cohorts, tx, logs, cutoff = _fixture_frames()
    out = _transaction_features(tx, cohorts)
    # most recent transaction is 2024-07-01, cutoff is 2024-07-15 -> 14 days
    assert int(out.loc["A", "recency_days"]) == 14


def test_features_do_not_perfectly_predict_label(feature_table):
    """A single feature that separates the classes almost perfectly is the
    signature of leakage. None should come close."""
    from sklearn.metrics import roc_auc_score
    y = feature_table["is_churn"]
    for col in NUMERIC_FEATURES:
        v = feature_table[col]
        if v.notna().sum() < 100 or v.nunique(dropna=True) < 2:
            continue
        auc = roc_auc_score(y, v.fillna(v.median()))
        auc = max(auc, 1 - auc)
        assert auc < 0.95, f"{col} alone separates the label (AUC={auc:.3f}) - likely leakage"
