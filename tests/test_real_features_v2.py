"""
Tests for the leakage-safe v2 feature build.

The important ones here are the cutoff tests. A leakage bug does not raise --
it silently produces a better score, which is exactly how the original 0.90 AUC
happened. So these assert on the discarded-row counters and the observed
maxima, not merely that the code ran.

Fixture-driven throughout; nothing touches the 1.6 GiB transactions file or the
30.5 GiB archive.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src.data import real_features_v2 as F


def tx(msno, trans, expire, cancel="0", auto="1", price="149", paid="149",
       days="30", method="41"):
    return {"msno": msno, "payment_method_id": method, "payment_plan_days": days,
            "plan_list_price": price, "actual_amount_paid": paid,
            "is_auto_renew": auto, "transaction_date": trans,
            "membership_expire_date": expire, "is_cancel": cancel}


@pytest.fixture
def population():
    return pd.Index(["a", "b", "c"], name="msno")


# ------------------------------------------------------------ cutoff constants

def test_cutoff_is_the_documented_observation_date():
    assert F.CUTOFF == "20170228"
    assert F.CUTOFF_DATE.isoformat() == "2017-02-28"


def test_windows_all_end_at_the_cutoff_and_are_ordered():
    """Every trailing window must start before the cutoff and none may reach
    past it. The previous-30 block must sit strictly before the last-30 block."""
    assert F.W7 == "20170222"    # 7 days inclusive of the cutoff
    assert F.W30 == "20170130"
    assert F.W90 == "20161201"
    assert F.P30_START == "20161231"
    assert F.P30_END == "20170129"
    assert F.P30_END < F.W30, "previous-30 must not overlap the last-30 window"
    for w in (F.W7, F.W30, F.W90, F.P30_START, F.P30_END):
        assert w <= F.CUTOFF


def test_forbidden_columns_include_the_label_and_raw_dates():
    assert "is_churn" in F.FORBIDDEN_COLUMNS
    assert "membership_expire_date" in F.FORBIDDEN_COLUMNS
    assert "transaction_date" in F.FORBIDDEN_COLUMNS


def test_transactions_v2_is_never_referenced_as_a_source():
    """March records are the label period; the module must not read them."""
    src = (F.TRANSACTIONS, F.MEMBERS, F.USER_LOGS_7Z, F.TRAIN_V2)
    assert not any("transactions_v2" in str(p) for p in src)


# --------------------------------------------------------- cutoff enforcement

def test_no_post_cutoff_transactions_reach_the_aggregates(tmp_path, population,
                                                          monkeypatch):
    """A March transaction must be discarded, not merely ignored downstream."""
    path = tmp_path / "transactions.csv"
    pd.DataFrame([
        tx("a", "20170115", "20170315"),
        tx("a", "20170301", "20170401"),   # after the cutoff
        tx("b", "20170228", "20170330"),   # exactly the cutoff: kept
    ], dtype=str).to_csv(path, index=False)
    monkeypatch.setattr(F, "TRANSACTIONS", path)

    out, stats = F.build_transaction_features(population)

    assert stats["rows_after_cutoff_discarded"] == 1
    assert stats["max_transaction_date_kept"] == "20170228"
    assert stats["max_transaction_date_kept"] <= F.CUTOFF
    # a has exactly one in-window transaction, not two
    assert out.loc["a", "transaction_count"] == 1
    assert out.loc["b", "transaction_count"] == 1


def test_cutoff_boundary_is_inclusive(tmp_path, population, monkeypatch):
    path = tmp_path / "transactions.csv"
    pd.DataFrame([tx("a", "20170228", "20170330")], dtype=str).to_csv(path, index=False)
    monkeypatch.setattr(F, "TRANSACTIONS", path)
    out, stats = F.build_transaction_features(population)
    assert stats["rows_after_cutoff_discarded"] == 0
    assert out.loc["a", "transaction_count"] == 1


def test_days_since_last_transaction_is_measured_from_the_cutoff(
        tmp_path, population, monkeypatch):
    path = tmp_path / "transactions.csv"
    pd.DataFrame([tx("a", "20170218", "20170320")], dtype=str).to_csv(path, index=False)
    monkeypatch.setattr(F, "TRANSACTIONS", path)
    out, _ = F.build_transaction_features(population)
    assert out.loc["a", "days_since_last_transaction"] == 10


def test_expiry_after_cutoff_is_allowed_and_positive(tmp_path, population,
                                                     monkeypatch):
    """membership_expire_date may fall past the cutoff -- it is the contract
    already held at the cutoff, not future information."""
    path = tmp_path / "transactions.csv"
    pd.DataFrame([tx("a", "20170201", "20170331")], dtype=str).to_csv(path, index=False)
    monkeypatch.setattr(F, "TRANSACTIONS", path)
    out, _ = F.build_transaction_features(population)
    assert out.loc["a", "days_until_membership_expiry"] == 31


def test_current_state_uses_the_comparator_not_the_max_expiry(
        tmp_path, population, monkeypatch):
    """A same-day cancellation supersedes the subscription it cancels."""
    path = tmp_path / "transactions.csv"
    pd.DataFrame([
        tx("a", "20170210", "20170610", cancel="0"),
        tx("a", "20170210", "20170310", cancel="1"),
    ], dtype=str).to_csv(path, index=False)
    monkeypatch.setattr(F, "TRANSACTIONS", path)
    out, _ = F.build_transaction_features(population)
    assert out.loc["a", "current_cancel_status"] == 1
    # 20170310 is 10 days past the cutoff, not 20170610 (102 days)
    assert out.loc["a", "days_until_membership_expiry"] == 10


def test_subscribers_without_transactions_are_nan_not_zero(
        tmp_path, population, monkeypatch):
    """Absence of a transaction is unknown, not a count of zero."""
    path = tmp_path / "transactions.csv"
    pd.DataFrame([tx("a", "20170115", "20170315")], dtype=str).to_csv(path, index=False)
    monkeypatch.setattr(F, "TRANSACTIONS", path)
    out, _ = F.build_transaction_features(population)
    assert out.loc["a", "transaction_count"] == 1
    assert np.isnan(out.loc["b", "transaction_count"])
    assert np.isnan(out.loc["c", "average_list_price"])


def test_aggregates_are_computed_over_the_window_only(tmp_path, population,
                                                      monkeypatch):
    path = tmp_path / "transactions.csv"
    pd.DataFrame([
        tx("a", "20160101", "20160201", price="100", paid="100"),
        tx("a", "20170215", "20170315", price="200", paid="150"),
        tx("a", "20170305", "20170405", price="999", paid="999"),  # discarded
    ], dtype=str).to_csv(path, index=False)
    monkeypatch.setattr(F, "TRANSACTIONS", path)
    out, _ = F.build_transaction_features(population)
    assert out.loc["a", "transaction_count"] == 2
    assert out.loc["a", "average_list_price"] == 150.0      # (100+200)/2
    assert out.loc["a", "mean_discount"] == 25.0            # (0+50)/2
    assert out.loc["a", "txns_last_30d"] == 1               # only 20170215


# ------------------------------------------------------------- member join

def test_member_join_and_tenure_relative_to_cutoff(tmp_path, population,
                                                   monkeypatch):
    path = tmp_path / "members.csv"
    pd.DataFrame([
        {"msno": "a", "city": "13", "bd": 29, "gender": "male",
         "registered_via": "9", "registration_init_time": "20170128"},
        {"msno": "b", "city": "1", "bd": -5, "gender": "",
         "registered_via": "7", "registration_init_time": "20150101"},
    ]).to_csv(path, index=False)
    monkeypatch.setattr(F, "MEMBERS", path)

    out, stats = F.build_member_features(population)

    assert out.loc["a", "tenure_days"] == 31
    assert out.loc["a", "age"] == 29
    assert out.loc["a", "city"] == "13"
    assert np.isnan(out.loc["b", "age"]), "out-of-range bd must be missing"
    assert stats["bd_out_of_range"] == 1
    assert out.loc["b", "gender"] == "unknown"
    assert "c" not in out.index, "unmatched population members are left to the join"


def test_registration_after_cutoff_is_flagged_not_negative(tmp_path, population,
                                                           monkeypatch):
    path = tmp_path / "members.csv"
    pd.DataFrame([{"msno": "a", "city": "1", "bd": 30, "gender": "female",
                   "registered_via": "7",
                   "registration_init_time": "20170401"}]).to_csv(path, index=False)
    monkeypatch.setattr(F, "MEMBERS", path)
    out, stats = F.build_member_features(population)
    assert stats["registration_after_cutoff"] == 1
    assert np.isnan(out.loc["a", "tenure_days"])
    assert out.loc["a", "registered_after_cutoff"] == 1


# ------------------------------------------------- label exclusion / stability

def test_no_feature_is_derived_from_the_label():
    """The label may be read only to attach is_churn, never inside a builder."""
    import inspect
    for fn in (F.build_member_features, F.build_transaction_features,
               F.build_userlog_features):
        assert "is_churn" not in inspect.getsource(fn)


def test_manifest_traces_every_feature_to_a_source_and_window():
    frame = pd.DataFrame({
        "city": ["1"], "age": [30.0], "transaction_count": [2.0],
        "log_total_secs": [10.0], "is_churn": [0],
    })
    manifest = F._manifest(frame, {"train_v2": "x"})
    assert "is_churn" not in manifest["features"]
    assert manifest["feature_count"] == 4
    assert manifest["observation_cutoff"] == "2017-02-28"
    assert manifest["features"]["log_total_secs"]["source"] == "user_logs.csv.7z"
    assert manifest["features"]["city"]["source"] == "members_v3.csv"
    assert manifest["features"]["transaction_count"]["source"] == "transactions.csv"
    for spec in manifest["features"].values():
        assert "2017-02-28" in spec["window"]
    assert "transactions_v2.csv" in manifest["excluded_sources"]
    json.dumps(manifest)  # must be serialisable


def test_validation_report_flags_a_post_cutoff_source():
    """If a max observed date ever exceeded the cutoff the report must say so."""
    frame = pd.DataFrame({"transaction_count": [1.0], "is_churn": [0]})
    labels = pd.DataFrame({"msno": ["a"], "is_churn": [0]})
    stats = {
        "transactions": {"max_transaction_date_kept": "20170301",
                         "rows_after_cutoff_discarded": 0},
        "user_logs": {"max_log_date_kept": 20170228,
                      "rows_after_cutoff_discarded": 0},
        "coverage": {},
    }
    report = F._validate(frame, labels, stats)
    checks = report["leakage_checks"]
    assert checks["transactions_within_cutoff"] is False
    assert checks["any_feature_sourced_after_cutoff"] is True


def test_validation_report_passes_a_clean_build():
    frame = pd.DataFrame({"transaction_count": [1.0, 2.0], "is_churn": [0, 1]})
    labels = pd.DataFrame({"msno": ["a", "b"], "is_churn": [0, 1]})
    stats = {
        "transactions": {"max_transaction_date_kept": "20170228",
                         "rows_after_cutoff_discarded": 7},
        "user_logs": {"max_log_date_kept": 20170228,
                      "rows_after_cutoff_discarded": 3},
        "coverage": {},
    }
    report = F._validate(frame, labels, stats)
    checks = report["leakage_checks"]
    assert checks["any_feature_sourced_after_cutoff"] is False
    assert checks["transactions_within_cutoff"] is True
    assert checks["user_logs_within_cutoff"] is True
    assert checks["label_columns_in_features"] == []
    assert report["label_distribution"] == {"0": 1, "1": 1}


def test_validation_report_detects_a_label_column_leaking_into_features():
    frame = pd.DataFrame({"transaction_count": [1.0],
                          "membership_expire_date": ["20170301"],
                          "is_churn": [0]})
    labels = pd.DataFrame({"msno": ["a"], "is_churn": [0]})
    stats = {"transactions": {"max_transaction_date_kept": "20170228",
                              "rows_after_cutoff_discarded": 0},
             "user_logs": {"max_log_date_kept": 20170228,
                           "rows_after_cutoff_discarded": 0},
             "coverage": {}}
    report = F._validate(frame, labels, stats)
    assert report["leakage_checks"]["label_columns_in_features"] == [
        "membership_expire_date"]
