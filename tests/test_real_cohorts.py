"""
Tests for the WSDMChurnLabeller port in src/data/real_cohorts.py.

These run on hand-built fixtures, not the 1.6 GiB Kaggle file, so they execute
in milliseconds and pin the behaviour that is easy to get wrong: the comparator
in calculateLastday. Each ordering case below mirrors a specific branch of the
Scala comparator and is named after the comment KKBOX wrote beside it.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.data import real_cohorts as rc


def tx(msno: str, trans: str, expire: str, cancel: str = "0",
       price: str = "149", days: str = "30", method: str = "41") -> dict[str, str]:
    """One transaction row, with the columns the labeller sorts on."""
    return {"msno": msno, "payment_method_id": method, "payment_plan_days": days,
            "plan_list_price": price, "transaction_date": trans,
            "membership_expire_date": expire, "is_cancel": cancel}


def frame(*rows: dict[str, str]) -> pd.DataFrame:
    return pd.DataFrame(list(rows), dtype=str)


# ------------------------------------------------------------ date/window logic

def test_history_window_constants_match_the_scala():
    """The official constants, hard-coded in WSDMChurnLabeller.scala."""
    assert rc.HISTORY_START == "20170101"
    assert rc.HISTORY_CUTOFF == "20170131"
    assert rc.CANDIDATE_START == "20170201"
    assert rc.CANDIDATE_END == "20170228"
    assert rc.OBSERVATION_CUTOFF.isoformat() == "2017-01-31"


@pytest.mark.parametrize("trans_date,in_history", [
    ("20161231", False),  # before the window
    ("20170101", True),   # inclusive lower bound
    ("20170115", True),
    ("20170131", True),   # inclusive upper bound -- cutoff is <=, not <
    ("20170201", False),  # future side
])
def test_history_window_is_inclusive_at_both_ends(trans_date, in_history):
    td = pd.Series([trans_date])
    got = bool(((td >= rc.HISTORY_START) & (td <= rc.HISTORY_CUTOFF)).iloc[0])
    assert got is in_history


@pytest.mark.parametrize("expire,selected", [
    ("20170131", False),  # expires before the window
    ("20170201", True),   # inclusive lower bound
    ("20170214", True),
    ("20170228", True),   # inclusive upper bound
    ("20170301", False),  # expires after the window
])
def test_candidate_expiry_window_is_inclusive(expire, selected):
    got = rc.select_candidates(
        pd.DataFrame({"msno": ["a"], "last_expire": [expire]}))
    assert (len(got) == 1) is selected


# ------------------------------------------------------ calculateLastday ordering

def test_lastday_is_not_simply_the_max_expiry():
    """A same-day cancellation sorts last and wins, though its expiry is lower.

    This is the case a naive max(membership_expire_date) gets wrong.
    """
    got = rc.calculate_lastday(frame(
        tx("a", "20170110", "20170410", cancel="0"),
        tx("a", "20170110", "20170210", cancel="1"),
    ))
    assert got.loc[0, "last_expire"] == "20170210"


def test_later_transaction_date_wins_regardless_of_expiry():
    got = rc.calculate_lastday(frame(
        tx("a", "20170120", "20170820", cancel="0"),
        tx("a", "20170125", "20170225", cancel="0"),
    ))
    assert got.loc[0, "last_expire"] == "20170225"


def test_consecutive_cancels_take_the_earlier_expiry():
    """"multiple cancel, consecutive cancels should only put the expiration
    date earlier" -- so descending expiry order, last one is the smallest."""
    got = rc.calculate_lastday(frame(
        tx("a", "20170110", "20170320", cancel="1"),
        tx("a", "20170110", "20170220", cancel="1"),
    ))
    assert got.loc[0, "last_expire"] == "20170220"


def test_consecutive_renewals_take_the_later_expiry():
    """"multiple renewal, expiration date keeps extending" -- ascending, so the
    last row holds the furthest expiry."""
    got = rc.calculate_lastday(frame(
        tx("a", "20170110", "20170320", cancel="0"),
        tx("a", "20170110", "20170220", cancel="0"),
    ))
    assert got.loc[0, "last_expire"] == "20170320"


def test_higher_plan_signature_sorts_first_so_lower_one_wins():
    """"same plan, always subscribe then unsubscribe" -- signature DESCENDING.

    Signature is the string concat price+days+method, so "99" + "30" + "41"
    gives "993041", which sorts above "1493041" as text. The lower signature
    therefore lands last.
    """
    got = rc.calculate_lastday(frame(
        tx("a", "20170110", "20170510", price="99", days="30", method="41"),
        tx("a", "20170110", "20170610", price="149", days="30", method="41"),
    ))
    assert rc.signature(frame(tx("a", "x", "y", price="99"))).iloc[0] == "993041"
    assert got.loc[0, "last_expire"] == "20170610"


def test_signature_is_string_concatenation_not_addition():
    sig = rc.signature(frame(tx("a", "20170110", "20170210",
                                price="129", days="30", method="41")))
    assert sig.iloc[0] == "1293041"


def test_subscription_precedes_cancellation_on_the_same_plan_and_day():
    """Identical signature, same day, mixed is_cancel: "0" sorts before "1"."""
    ordered = rc.order_transactions(frame(
        tx("a", "20170110", "20170210", cancel="1"),
        tx("a", "20170110", "20170410", cancel="0"),
    ))
    assert list(ordered["is_cancel"]) == ["0", "1"]


# ------------------------------------------------------------- duplicate handling

def test_multiple_subscribers_are_grouped_independently():
    got = rc.calculate_lastday(frame(
        tx("a", "20170105", "20170205"),
        tx("a", "20170120", "20170220"),
        tx("b", "20170103", "20170303"),
    )).set_index("msno")["last_expire"].to_dict()
    assert got == {"a": "20170220", "b": "20170303"}


def test_one_row_per_subscriber_even_with_many_transactions():
    rows = [tx("a", f"201701{d:02d}", f"201702{d:02d}") for d in range(1, 29)]
    got = rc.calculate_lastday(frame(*rows))
    assert len(got) == 1
    assert got.loc[0, "msno"] == "a"
    # The latest transaction_date wins.
    assert got.loc[0, "last_expire"] == "20170228"


def test_exact_duplicate_rows_collapse_to_one_subscriber():
    row = tx("a", "20170110", "20170210")
    got = rc.calculate_lastday(frame(row, row, row))
    assert len(got) == 1
    assert got.loc[0, "last_expire"] == "20170210"


def test_candidate_selection_preserves_one_row_per_subscriber():
    last = pd.DataFrame({"msno": ["a", "b", "c"],
                         "last_expire": ["20170210", "20170305", "20170228"]})
    got = rc.select_candidates(last)
    assert list(got["msno"]) == ["a", "c"]
    assert got["msno"].is_unique


# --------------------------------------------------------- invalid/sentinel expiry

def test_sentinel_expiry_is_excluded_from_the_candidate_window():
    """19700101 is KKBOX's null marker; it must not be read as a real date."""
    got = rc.select_candidates(
        pd.DataFrame({"msno": ["a"], "last_expire": [rc.SENTINEL_EXPIRY]}))
    assert len(got) == 0


def test_sentinel_expiry_survives_lastday_so_it_can_be_counted():
    """The port does not drop sentinel rows silently -- they flow through and
    are reported, which is what lets the validation report count them."""
    got = rc.calculate_lastday(frame(
        tx("a", "20170110", "20170410", cancel="0"),
        tx("a", "20170115", rc.SENTINEL_EXPIRY, cancel="1"),
    ))
    assert got.loc[0, "last_expire"] == rc.SENTINEL_EXPIRY


def test_sentinel_as_cancel_sorts_by_its_numeric_value():
    """A 19700101 cancellation is numerically the earliest possible expiry, so
    among consecutive cancels (descending expiry) it lands last."""
    got = rc.calculate_lastday(frame(
        tx("a", "20170110", "20170310", cancel="1"),
        tx("a", "20170110", rc.SENTINEL_EXPIRY, cancel="1"),
    ))
    assert got.loc[0, "last_expire"] == rc.SENTINEL_EXPIRY


def test_empty_history_yields_no_candidates():
    got = rc.calculate_lastday(pd.DataFrame(columns=rc.READ_COLUMNS, dtype=str))
    assert len(got) == 0
    assert len(rc.select_candidates(got)) == 0


# ------------------------------------------------------------- renewal gap port

def test_renewal_gap_counts_days_from_expiry_to_renewal():
    gap = rc.calculate_renewal_gap(
        [tx("a", "20170215", "20170315", cancel="0")], "20170210")
    assert gap == 5


def test_renewal_gap_is_9999_when_no_renewal_follows():
    gap = rc.calculate_renewal_gap(
        [tx("a", "20170215", "20170201", cancel="1")], "20170210")
    assert gap == 9999


def test_cancellation_pulls_the_expiry_earlier_before_the_gap_is_measured():
    """A cancel that moves the expiry back makes the later renewal look worse."""
    gap = rc.calculate_renewal_gap([
        tx("a", "20170201", "20170205", cancel="1"),   # expiry 0210 -> 0205
        tx("a", "20170215", "20170315", cancel="0"),
    ], "20170210")
    assert gap == 10


def test_cancellation_never_pushes_the_expiry_later():
    gap = rc.calculate_renewal_gap([
        tx("a", "20170201", "20170401", cancel="1"),   # later: ignored
        tx("a", "20170215", "20170315", cancel="0"),
    ], "20170210")
    assert gap == 5


# ------------------------------------------------- end-to-end validation contract

def test_build_cohorts_reports_exact_match_on_a_consistent_fixture(tmp_path):
    """Reconstruction and train.csv agreeing exactly -> EXACT_MATCH."""
    transactions = tmp_path / "transactions.csv"
    frame(
        tx("a", "20170110", "20170210"),          # candidate
        tx("b", "20170112", "20170215"),          # candidate
        tx("c", "20170112", "20170401"),          # expires too late
        tx("d", "20161201", "20170210"),          # outside history window
    ).to_csv(transactions, index=False)

    train = tmp_path / "train.csv"
    pd.DataFrame({"msno": ["a", "b"], "is_churn": [1, 0]}).to_csv(train, index=False)

    cohorts, report = build(tmp_path, train, transactions)

    assert report["validation_status"] == "EXACT_MATCH"
    assert report["train_rows"] == 2
    assert report["reconstructed_candidate_rows"] == 2
    assert report["intersection_rows"] == 2
    assert report["train_missing_from_candidates"] == 0
    assert report["candidates_missing_from_train"] == 0
    assert report["label_distribution"]["cohort"] == {"0": 1, "1": 1}
    assert set(cohorts.columns) == {"msno", "observation_cutoff", "last_expire",
                                    "cohort", "is_churn"}
    assert cohorts["cohort"].unique().tolist() == ["2017-02"]
    assert cohorts["observation_cutoff"].dt.date.unique().tolist() == [
        rc.OBSERVATION_CUTOFF]


def test_build_cohorts_reports_mismatch_without_forcing_agreement(tmp_path):
    """A subscriber in train.csv that the reconstruction misses must surface as
    MISMATCH, not be quietly dropped."""
    transactions = tmp_path / "transactions.csv"
    frame(tx("a", "20170110", "20170210")).to_csv(transactions, index=False)

    train = tmp_path / "train.csv"
    pd.DataFrame({"msno": ["a", "ghost"], "is_churn": [1, 0]}).to_csv(
        train, index=False)

    _cohorts, report = build(tmp_path, train, transactions)

    assert report["validation_status"] == "MISMATCH"
    assert report["train_missing_from_candidates"] == 1
    assert report["candidates_missing_from_train"] == 0


def test_build_cohorts_counts_sentinel_expiries(tmp_path):
    transactions = tmp_path / "transactions.csv"
    frame(
        tx("a", "20170110", "20170210"),
        tx("b", "20170110", rc.SENTINEL_EXPIRY),
    ).to_csv(transactions, index=False)

    train = tmp_path / "train.csv"
    pd.DataFrame({"msno": ["a"], "is_churn": [1]}).to_csv(train, index=False)

    _cohorts, report = build(tmp_path, train, transactions)

    invalid = report["invalid_expiry_count"]
    assert invalid["sentinel_19700101_in_transactions"] == 1
    assert invalid["sentinel_19700101_as_last_expire"] == 1
    # b is excluded from candidates, so the cohort still matches train exactly.
    assert report["validation_status"] == "EXACT_MATCH"


def build(tmp_path, train, transactions):
    """Run build_cohorts against a fixture, writing into tmp_path."""
    return rc.build_cohorts(
        train_csv=train, transactions_csv=transactions,
        out_parquet=tmp_path / "cohorts.parquet",
        out_report=tmp_path / "report.json",
        write=True)
