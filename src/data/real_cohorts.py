"""
Stage 1 of the real-KKBOX migration: reconstruct the official prediction cohort.

This is a faithful port of WSDMChurnLabeller.scala, the labeller KKBOX shipped
with the WSDM Cup 2018 churn competition. It does not simplify the tie-breaking
rules, because those rules are the whole difficulty: a subscriber's "last
expiration" is not max(membership_expire_date), it is the expiry of whichever
transaction sorts last under a comparator that encodes KKBOX's own assumptions
about same-day subscribe/cancel pairs.

What this module does NOT do: assign labels. The reconstruction produces the
candidate *set*; `is_churn` is joined from the official train.csv. Two reasons.
First, the task requires ground truth to come from KKBOX. Second, it could not
be done honestly here anyway -- transactions.csv ends 2017-02-28, so a renewal
in March is invisible and calculate_renewal_gap() would mislabel late renewals
as churn. The gap function is ported and tested for fidelity, but is not used
as a label source.

The raw data lives outside the repository and is never copied into it; only the
derived cohort table is written to data/processed/real/.

Reference: https://www.kaggle.com/c/kkbox-churn-prediction-challenge
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

log = logging.getLogger("real_cohorts")

# --------------------------------------------------------------- locations
# Overridable so the module is testable and portable; the default matches where
# the Kaggle download currently sits. Raw data is READ ONLY and never copied.
KKBOX_DIR = Path(os.environ.get(
    "CHURNCAST_KKBOX_DIR", r"C:/Users/Sanjay kanth/data/raw_kkbox"))

TRAIN_CSV = KKBOX_DIR / "train" / "train.csv"
TRANSACTIONS_CSV = KKBOX_DIR / "transactions" / "transactions.csv"

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "processed" / "real"
COHORTS_PARQUET = OUT_DIR / "cohorts.parquet"
VALIDATION_JSON = OUT_DIR / "cohort_validation_report.json"

# ---------------------------------------------------- official window constants
#
# Straight from the Scala:
#     val historyCutoff = "20170131"
#     historyData = data.filter(transaction_date >= "20170101"
#                               and transaction_date <= historyCutoff)
#     futureData  = data.filter(transaction_date >  historyCutoff)
#     predictionCandidates = userExpire.filter(
#         last_expire >= "20170201" and last_expire <= "20170228")
#
# Dates are compared as STRINGS throughout, exactly as Spark did on a CSV read
# with no schema. For fixed-width yyyyMMdd this orders identically to integers,
# but the string form is kept so the port cannot drift from the original.
HISTORY_START = "20170101"
HISTORY_CUTOFF = "20170131"
CANDIDATE_START = "20170201"
CANDIDATE_END = "20170228"

COHORT = "2017-02"
OBSERVATION_CUTOFF = date(2017, 1, 31)

# Columns calculateLastday sorts on. Read as str: the sort signature is a raw
# string concatenation of the CSV text, so "129"+"30"+"41" -> "1293041". Parsing
# these as numbers first would silently change the ordering.
SORT_COLUMNS = ["payment_method_id", "payment_plan_days", "plan_list_price",
                "transaction_date", "membership_expire_date", "is_cancel"]
READ_COLUMNS = ["msno"] + SORT_COLUMNS

# KKBOX uses 19700101 (the Unix epoch) as a null/sentinel expiry rather than a
# real expiration date. Counted and reported, never silently treated as a date.
SENTINEL_EXPIRY = "19700101"

CHUNK_SIZE = 2_000_000


# --------------------------------------------------------------- port: ordering

def signature(frame: pd.DataFrame) -> pd.Series:
    """The Scala's plan signature: plan_list_price + payment_plan_days + method.

        val x_sig = x.plan_list_price + x.payment_plan_days + x.payment_method_id

    String concatenation, not addition. Two transactions share a signature only
    when all three fields match as text.
    """
    return (frame["plan_list_price"].astype(str)
            + frame["payment_plan_days"].astype(str)
            + frame["payment_method_id"].astype(str))


def _expiry_key(frame: pd.DataFrame) -> pd.Series:
    """Collapse the cancel-dependent expiry ordering into one ascending key.

    The comparator orders tied rows by membership_expire_date *descending* when
    both are cancellations (consecutive cancels only pull the expiry earlier)
    and *ascending* when both are renewals (renewals keep extending it). Negating
    the cancel case lets a single ascending sort reproduce both, which matters:
    the rows being compared always share an is_cancel value by this point,
    because is_cancel is a higher-priority key.
    """
    expiry = pd.to_numeric(frame["membership_expire_date"], errors="coerce").fillna(0)
    cancelled = frame["is_cancel"].astype(str) == "1"
    return expiry.where(~cancelled, -expiry)


def order_transactions(frame: pd.DataFrame) -> pd.DataFrame:
    """Sort exactly as WSDMChurnLabeller's comparator does.

    The Scala comparator, in priority order:
      1. transaction_date ascending
      2. plan signature DESCENDING   ("same plan, always subscribe then
         unsubscribe" -- the higher signature sorts first)
      3. is_cancel ascending          (subscription precedes cancellation)
      4. membership_expire_date: ascending for renewals, descending for cancels
    """
    work = frame.copy()
    work["_sig"] = signature(work)
    work["_expiry_key"] = _expiry_key(work)
    return work.sort_values(
        ["msno", "transaction_date", "_sig", "is_cancel", "_expiry_key"],
        ascending=[True, True, False, True, True],
        kind="mergesort",  # stable, so equal rows keep input order
    )


def calculate_lastday(history: pd.DataFrame) -> pd.DataFrame:
    """Port of calculateLastday: per-subscriber last expiration in the history.

    The Scala groups by msno, orders the group with the comparator above and
    takes `orderedList.last.membership_expire_date`. Note what this is NOT: it
    is not max(membership_expire_date). A same-day cancellation sorts after its
    subscription and therefore wins, which is the entire point of the ordering.

    Returns a frame of msno -> last_expire (as the original yyyyMMdd string).
    """
    if history.empty:
        return pd.DataFrame({"msno": [], "last_expire": []}, dtype=str)

    ordered = order_transactions(history)
    last = ordered.groupby("msno", sort=False).tail(1)
    return (last[["msno", "membership_expire_date"]]
            .rename(columns={"membership_expire_date": "last_expire"})
            .reset_index(drop=True))


def select_candidates(last_expire: pd.DataFrame,
                      start: str = CANDIDATE_START,
                      end: str = CANDIDATE_END) -> pd.DataFrame:
    """Port of the predictionCandidates filter: last_expire within the window.

    Compared as strings, as in the Scala.
    """
    expire = last_expire["last_expire"].astype(str)
    keep = (expire >= start) & (expire <= end)
    return last_expire.loc[keep].reset_index(drop=True)


# --------------------------------------------------- port: renewal gap (unused)

def calculate_renewal_gap(rows: Iterable[dict[str, str]], last_expiration: str) -> int:
    """Port of calculateRenewalGap. Ported for fidelity; NOT used for labelling.

    Walks the ordered future transactions looking for the first non-cancel
    action. Cancellations before it only pull the effective expiry earlier. The
    returned gap is the day count from that (possibly moved) expiry to the
    renewal's transaction_date; 9999 means no renewal was found at all.

    The Scala then treats gap >= 30 as churn. That is deliberately not applied
    here: transactions.csv stops at 2017-02-28, so a March renewal is invisible
    and would be scored as churn. Labels come from train.csv instead.
    """
    def parse(value: str) -> datetime:
        return datetime.strptime(value, "%Y%m%d")

    last_expire_date = parse(last_expiration)
    gap = 9999
    for row in rows:
        if gap != 9999:
            break
        if str(row["is_cancel"]) == "1":
            expire_date = parse(str(row["membership_expire_date"]))
            if expire_date < last_expire_date:
                last_expire_date = expire_date
        else:
            trans_date = parse(str(row["transaction_date"]))
            gap = (trans_date - last_expire_date).days
    return gap


# --------------------------------------------------------------- data loading

def load_transaction_windows(
    path: Path = TRANSACTIONS_CSV,
    history_start: str = HISTORY_START,
    history_cutoff: str = HISTORY_CUTOFF,
    chunk_size: int = CHUNK_SIZE,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Stream transactions.csv, keeping only the two windows the labeller needs.

    The file is ~21.5M rows / 1.6 GiB, so it is never loaded whole: each chunk
    is filtered down to the January history window and the post-cutoff future
    window before being retained. Everything is read as str so the sort
    signature matches the original CSV text.
    """
    history_parts: list[pd.DataFrame] = []
    future_parts: list[pd.DataFrame] = []
    stats = {"total_rows": 0, "sentinel_expiry_rows": 0,
             "transaction_date_min": None, "transaction_date_max": None}

    for chunk in pd.read_csv(path, usecols=READ_COLUMNS, dtype=str,
                             chunksize=chunk_size):
        stats["total_rows"] += len(chunk)
        stats["sentinel_expiry_rows"] += int(
            (chunk["membership_expire_date"] == SENTINEL_EXPIRY).sum())

        lo, hi = chunk["transaction_date"].min(), chunk["transaction_date"].max()
        stats["transaction_date_min"] = (
            lo if stats["transaction_date_min"] is None
            else min(stats["transaction_date_min"], lo))
        stats["transaction_date_max"] = (
            hi if stats["transaction_date_max"] is None
            else max(stats["transaction_date_max"], hi))

        td = chunk["transaction_date"]
        history_parts.append(chunk[(td >= history_start) & (td <= history_cutoff)])
        future_parts.append(chunk[td > history_cutoff])

    history = (pd.concat(history_parts, ignore_index=True)
               if history_parts else pd.DataFrame(columns=READ_COLUMNS))
    future = (pd.concat(future_parts, ignore_index=True)
              if future_parts else pd.DataFrame(columns=READ_COLUMNS))
    log.info("transactions: %s rows -> history %s, future %s",
             f"{stats['total_rows']:,}", f"{len(history):,}", f"{len(future):,}")
    return history, future, stats


# --------------------------------------------------------------- reconstruction

def build_cohorts(
    train_csv: Path = TRAIN_CSV,
    transactions_csv: Path = TRANSACTIONS_CSV,
    out_parquet: Path = COHORTS_PARQUET,
    out_report: Path = VALIDATION_JSON,
    write: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reconstruct the official cohort and validate it against train.csv.

    Returns (cohorts, report). The cohort table carries the official label; the
    report records exactly how the reconstruction compares to ground truth,
    including any discrepancy. No attempt is made to force agreement.
    """
    train = pd.read_csv(train_csv, dtype={"msno": str, "is_churn": "int8"})
    history, _future, tx_stats = load_transaction_windows(transactions_csv)

    last_expire = calculate_lastday(history)
    candidates = select_candidates(last_expire)

    train_ids = set(train["msno"])
    cand_ids = set(candidates["msno"])
    intersection = train_ids & cand_ids

    cohorts = candidates.merge(train, on="msno", how="inner")
    cohorts["observation_cutoff"] = pd.Timestamp(OBSERVATION_CUTOFF)
    cohorts["cohort"] = COHORT
    cohorts["last_expire"] = pd.to_datetime(cohorts["last_expire"], format="%Y%m%d")
    cohorts = cohorts[["msno", "observation_cutoff", "last_expire",
                       "cohort", "is_churn"]]

    exact = (len(train_ids) == len(cand_ids) == len(intersection))

    # When the sets disagree, say why. Without this the report states only that
    # something is wrong, and the natural suspicion falls on the comparator --
    # which is measurably not the cause.
    missing_ids = train_ids - cand_ids
    history_ids = set(history["msno"])
    last_map = last_expire.set_index("msno")["last_expire"]
    missing_with_history = missing_ids & history_ids
    missing_expiry = last_map[last_map.index.isin(missing_with_history)]
    naive_max = history.groupby("msno")["membership_expire_date"].max()
    comparator_disagreements = int(
        (last_map != naive_max.reindex(last_map.index)).sum())

    discrepancy = {
        "missing_without_history_transaction": int(len(missing_ids - history_ids)),
        "missing_with_history_transaction": int(len(missing_with_history)),
        "missing_last_expire_before_window": int((missing_expiry < CANDIDATE_START).sum()),
        "missing_last_expire_after_window": int((missing_expiry > CANDIDATE_END).sum()),
        "comparator_vs_naive_max_disagreements": comparator_disagreements,
        "comparator_disagreement_rate": round(
            comparator_disagreements / max(len(last_map), 1), 6),
        "churn_rate_train": round(float(train["is_churn"].mean()), 6),
        "churn_rate_cohort": round(float(cohorts["is_churn"].mean()), 6)
        if len(cohorts) else None,
        "churn_rate_missing": round(
            float(train.loc[train["msno"].isin(missing_ids), "is_churn"].mean()), 6)
        if missing_ids else None,
        "note": (
            "The Scala reads wsdm_transactions_20170331.csv, a fixed snapshot "
            "extending to 2017-03-31. The published transactions.csv covers "
            "2015-01-01..2017-02-28 only, so subscribers whose membership was "
            "established outside the January window cannot be recovered from it. "
            "The comparator disagreement rate above shows the tie-breaking "
            "logic is exercised and is not the source of the gap."
        ),
    }
    report: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source": {
            "train_csv": str(train_csv),
            "transactions_csv": str(transactions_csv),
            "labeller": "WSDMChurnLabeller.scala (ported)",
        },
        "windows": {
            "history_start": HISTORY_START,
            "history_cutoff": HISTORY_CUTOFF,
            "candidate_start": CANDIDATE_START,
            "candidate_end": CANDIDATE_END,
            "cohort": COHORT,
            "observation_cutoff": OBSERVATION_CUTOFF.isoformat(),
        },
        "train_rows": int(len(train)),
        "train_unique_msno": int(len(train_ids)),
        "reconstructed_candidate_rows": int(len(candidates)),
        "reconstructed_unique_msno": int(len(cand_ids)),
        "intersection_rows": int(len(intersection)),
        "train_missing_from_candidates": int(len(train_ids - cand_ids)),
        "candidates_missing_from_train": int(len(cand_ids - train_ids)),
        "label_distribution": {
            "train": {str(k): int(v)
                      for k, v in train["is_churn"].value_counts().sort_index().items()},
            "cohort": {str(k): int(v)
                       for k, v in cohorts["is_churn"].value_counts().sort_index().items()},
        },
        "duplicate_msno_count": {
            "train": int(len(train) - train["msno"].nunique()),
            "candidates": int(len(candidates) - candidates["msno"].nunique()),
            "history_transactions_per_msno_max": int(
                history.groupby("msno").size().max()) if len(history) else 0,
        },
        "invalid_expiry_count": {
            "sentinel_19700101_in_transactions": int(tx_stats["sentinel_expiry_rows"]),
            "sentinel_19700101_as_last_expire": int(
                (last_expire["last_expire"] == SENTINEL_EXPIRY).sum()),
            "unparseable_last_expire": int(
                pd.to_datetime(last_expire["last_expire"], format="%Y%m%d",
                               errors="coerce").isna().sum()),
        },
        "date_ranges": {
            "transactions_transaction_date": [tx_stats["transaction_date_min"],
                                              tx_stats["transaction_date_max"]],
            "history_transaction_date": [
                history["transaction_date"].min() if len(history) else None,
                history["transaction_date"].max() if len(history) else None],
            "reconstructed_last_expire": [
                last_expire["last_expire"].min() if len(last_expire) else None,
                last_expire["last_expire"].max() if len(last_expire) else None],
            "candidate_last_expire": [
                candidates["last_expire"].min() if len(candidates) else None,
                candidates["last_expire"].max() if len(candidates) else None],
        },
        "transaction_rows_scanned": int(tx_stats["total_rows"]),
        "validation_status": "EXACT_MATCH" if exact else "MISMATCH",
        "discrepancy_analysis": None if exact else discrepancy,
    }

    if write:
        out_parquet.parent.mkdir(parents=True, exist_ok=True)
        cohorts.to_parquet(out_parquet, index=False)
        out_report.write_text(json.dumps(report, indent=2), encoding="utf-8")
        log.info("wrote %s (%s rows) and %s", out_parquet, f"{len(cohorts):,}",
                 out_report)

    return cohorts, report


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kkbox-dir", type=Path, default=KKBOX_DIR)
    parser.add_argument("--dry-run", action="store_true",
                        help="reconstruct and validate without writing files")
    args = parser.parse_args()

    train_csv = args.kkbox_dir / "train" / "train.csv"
    transactions_csv = args.kkbox_dir / "transactions" / "transactions.csv"
    for path in (train_csv, transactions_csv):
        if not path.exists():
            parser.error(f"missing required input: {path}")

    _cohorts, report = build_cohorts(train_csv, transactions_csv,
                                     write=not args.dry_run)

    print(json.dumps({k: report[k] for k in (
        "train_rows", "reconstructed_candidate_rows", "intersection_rows",
        "train_missing_from_candidates", "candidates_missing_from_train",
        "label_distribution", "validation_status")}, indent=2))

    # Requirement: do not silently force a match.
    return 0 if report["validation_status"] == "EXACT_MATCH" else 1


if __name__ == "__main__":
    raise SystemExit(main())
