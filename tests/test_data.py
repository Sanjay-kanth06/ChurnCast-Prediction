"""Synthetic dataset generation and raw-data integrity."""
from __future__ import annotations

import pandas as pd
import pytest

from src import config

EXPECTED_COLUMNS = {
    "members": {"msno", "city", "bd", "gender", "registered_via", "registration_init_time"},
    "transactions": {"msno", "payment_method_id", "payment_plan_days", "plan_list_price",
                     "actual_amount_paid", "is_auto_renew", "transaction_date",
                     "membership_expire_date", "is_cancel"},
    "user_logs": {"msno", "date", "num_25", "num_50", "num_75", "num_985",
                  "num_100", "num_unq", "total_secs"},
    "train": {"msno", "is_churn"},
}


def test_generator_is_deterministic(tmp_path, monkeypatch):
    """Same seed -> identical output."""
    import importlib
    import scripts.generate_sample_data as gen

    monkeypatch.setattr(config, "RAW_DIR", tmp_path)
    monkeypatch.setattr(config, "RAW_FILES",
                        {k: tmp_path / f"{k}.csv" for k in EXPECTED_COLUMNS})
    importlib.reload(gen)
    monkeypatch.setattr(gen.config, "RAW_DIR", tmp_path)
    monkeypatch.setattr(gen.config, "RAW_FILES",
                        {k: tmp_path / f"{k}.csv" for k in EXPECTED_COLUMNS})

    a = gen.generate(n_subscribers=200, seed=7)
    b = gen.generate(n_subscribers=200, seed=7)
    pd.testing.assert_frame_equal(a["train"], b["train"])
    pd.testing.assert_frame_equal(a["members"], b["members"])
    assert len(a["transactions"]) == len(b["transactions"])


@pytest.mark.parametrize("name", list(EXPECTED_COLUMNS))
def test_required_columns_present(raw_tables, name):
    assert EXPECTED_COLUMNS[name].issubset(set(raw_tables[name].columns))


@pytest.mark.parametrize("name", list(EXPECTED_COLUMNS))
def test_tables_not_empty(raw_tables, name):
    assert len(raw_tables[name]) > 0


def test_msno_relationships_are_valid(raw_tables):
    """Every msno in the event tables must exist in the label table."""
    labelled = set(raw_tables["train"]["msno"])
    assert set(raw_tables["members"]["msno"]).issubset(labelled)
    assert set(raw_tables["transactions"]["msno"]).issubset(labelled)
    assert set(raw_tables["user_logs"]["msno"]).issubset(labelled)


def test_msno_is_unique_in_subscriber_tables(raw_tables):
    assert raw_tables["train"]["msno"].is_unique
    assert raw_tables["members"]["msno"].is_unique


def test_msno_format(raw_tables):
    s = raw_tables["train"]["msno"]
    assert s.str.match(rf"^{config.MSNO_PREFIX}\d{{{config.MSNO_WIDTH}}}$").all()


def test_label_is_binary(raw_tables):
    assert set(raw_tables["train"]["is_churn"].unique()).issubset({0, 1})


def test_churn_rate_is_plausible(raw_tables):
    """Not degenerate in either direction - a usable classification problem."""
    rate = raw_tables["train"]["is_churn"].mean()
    assert 0.05 < rate < 0.60, f"implausible churn rate {rate:.3f}"


def test_no_unexpected_missing_values_in_raw(raw_tables):
    """Only members.gender may be blank -- the generator leaves it empty for a
    share of subscribers, mirroring KKBOX, and pandas reads "" back as NaN."""
    allowed = {("members", "gender")}
    for name, df in raw_tables.items():
        for col, n in df.isna().sum().items():
            if n and (name, col) not in allowed:
                pytest.fail(f"{name}.{col} has {n} nulls")


def test_subscribers_have_multiple_records(raw_tables):
    tx_counts = raw_tables["transactions"].groupby("msno").size()
    log_counts = raw_tables["user_logs"].groupby("msno").size()
    assert tx_counts.median() > 1
    assert log_counts.median() > 10


def test_cohorts_span_multiple_months(raw_tables):
    months = raw_tables["cohorts"]["cohort_month"].unique()
    assert len(months) >= 3, "temporal split needs several cohorts"


def test_renewal_transactions_fall_after_cutoff(raw_tables):
    """The label-defining renewal must be outside the feature window -- this is
    what makes the leakage guard meaningful rather than decorative."""
    tx = raw_tables["transactions"]
    cohorts = raw_tables["cohorts"].set_index("msno")["observation_cutoff"]
    labels = raw_tables["train"].set_index("msno")["is_churn"]

    stayers = labels[labels == 0].index[:200]
    merged = tx[tx["msno"].isin(stayers)].copy()
    merged["cutoff"] = merged["msno"].map(cohorts)
    after = merged[merged["transaction_date"] > merged["cutoff"]]
    assert len(after) > 0, "non-churners should have a renewal after their cutoff"
