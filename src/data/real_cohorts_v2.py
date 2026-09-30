"""
Stage 1b: reconstruct the KKBOX *v2* (March 2017) prediction cohort.

Separate from real_cohorts.py, which is left untouched, because the v2 round is
not the v1 round with different dates -- the evidence says the history rule
itself differs.

WHAT THE SOURCE ACTUALLY SUPPORTS
---------------------------------
KKBOX shipped exactly one labeller, WSDMChurnLabeller.scala, and it sits
*outside* both churn_comp_refresh/ folders. The refresh release (train_v2.csv,
transactions_v2.csv) carries no labeller, no README and no documentation. So
there is no source-level statement of the v2 labelling rule, and the v1 Scala
cannot simply be assumed to carry over. Two things are inferred from the data
rather than read from a spec, and are labelled as such throughout:

  * the candidate window. Not assumed: measured. Of the train_v2 members with a
    February 2017 transaction, 98.4% have a reconstructed last expiry in March
    2017 (835,415 of 849,357). The window is unambiguous.

  * the history rule. The v1 Scala bounds history at BOTH ends
    (transaction_date >= 20170101 AND <= 20170131). Applied literally to v2
    (a February window) that recovers 86.0% of train_v2. Dropping the lower
    bound -- every transaction up to the cutoff, which is what "observation
    cutoff" normally means -- recovers 97.55%. The narrow window is what the
    Scala says; the cutoff is what the data fits. Both are computed and both
    are reported, and the difference is not papered over.

The plausible reconciliation is that the Scala's lower bound was harmless
against its own input, wsdm_transactions_20170331.csv, and only becomes
destructive against the full 21.5M-row public transactions.csv, which reaches
back to 2015 and so contains the older memberships the lower bound discards.
That remains a hypothesis: the labeller's input file is not available.

Labels are never invented -- is_churn is joined from train_v2.csv.
Raw data is read from outside the repository and never copied into it.
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import date, datetime
from pathlib import Path

import pandas as pd

# The comparator is imported, not reimplemented and not modified. Its
# tie-breaking is a property of KKBOX's transaction semantics, which the refresh
# release did not change.
from src.data.real_cohorts import (
    KKBOX_DIR,
    READ_COLUMNS,
    SENTINEL_EXPIRY,
    order_transactions,
    select_candidates,
)

log = logging.getLogger("real_cohorts_v2")

TRAIN_V2_CSV = KKBOX_DIR / "train_v2" / "data" / "churn_comp_refresh" / "train_v2.csv"
TRANSACTIONS_V2_CSV = (KKBOX_DIR / "transactions_v2" / "data" / "churn_comp_refresh"
                       / "transactions_v2.csv")
TRANSACTIONS_V1_CSV = KKBOX_DIR / "transactions" / "transactions.csv"

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "processed" / "real_v2"
COHORTS_PARQUET = OUT_DIR / "cohorts_v2.parquet"
VALIDATION_JSON = OUT_DIR / "cohort_validation_report_v2.json"

# Inferred from the data (see module docstring), not read from a labeller.
HISTORY_CUTOFF = "20170228"
HISTORY_WINDOW_START = "20170201"   # the literal v1-style lower bound
CANDIDATE_START = "20170301"
CANDIDATE_END = "20170331"

COHORT = "2017-03"
OBSERVATION_CUTOFF = date(2017, 2, 28)

CHUNK_SIZE = 3_000_000


def lastday_to_cutoff(paths: list[Path], cutoff: str,
                      start: str | None = None,
                      chunk_size: int = CHUNK_SIZE) -> tuple[pd.DataFrame, dict]:
    """Per-subscriber last expiry at `cutoff`, folded chunk by chunk.

    calculate_lastday() in real_cohorts.py sorts one in-memory frame, which is
    fine for a single month. A full-history cutoff spans 21.5M rows, so this
    computes the same value as a reduction instead: the comparator defines a
    total order, so the per-subscriber maximum under it is associative and can
    be folded across chunks without ever materialising the whole file.
    """
    winners: pd.DataFrame | None = None
    stats = {"rows_scanned": 0, "rows_in_window": 0, "sentinel_rows": 0,
             "transaction_date_min": None, "transaction_date_max": None}

    for path in paths:
        for chunk in pd.read_csv(path, usecols=READ_COLUMNS, dtype=str,
                                 chunksize=chunk_size):
            stats["rows_scanned"] += len(chunk)
            td = chunk["transaction_date"]
            lo, hi = td.min(), td.max()
            stats["transaction_date_min"] = (
                lo if stats["transaction_date_min"] is None
                else min(stats["transaction_date_min"], lo))
            stats["transaction_date_max"] = (
                hi if stats["transaction_date_max"] is None
                else max(stats["transaction_date_max"], hi))

            mask = (td <= cutoff) if start is None else ((td >= start) & (td <= cutoff))
            chunk = chunk[mask]
            if chunk.empty:
                continue
            stats["rows_in_window"] += len(chunk)
            stats["sentinel_rows"] += int(
                (chunk["membership_expire_date"] == SENTINEL_EXPIRY).sum())

            pool = chunk if winners is None else pd.concat([winners, chunk],
                                                           ignore_index=True)
            winners = (order_transactions(pool)
                       .groupby("msno", sort=False).tail(1)[READ_COLUMNS]
                       .reset_index(drop=True))

    if winners is None:
        return pd.DataFrame({"msno": [], "last_expire": []}, dtype=str), stats

    last = order_transactions(winners).groupby("msno", sort=False).tail(1)
    out = (last[["msno", "membership_expire_date"]]
           .rename(columns={"membership_expire_date": "last_expire"})
           .reset_index(drop=True))
    return out, stats


def _compare(train_ids: set[str], cand_ids: set[str]) -> dict:
    inter = train_ids & cand_ids
    return {
        "candidates": len(cand_ids),
        "intersection": len(inter),
        "train_only": len(train_ids - cand_ids),
        "candidate_only": len(cand_ids - train_ids),
        "coverage_of_train_v2": round(len(inter) / max(len(train_ids), 1), 6),
        "exact_match": train_ids == cand_ids,
    }


def build_v2(write: bool = True) -> tuple[pd.DataFrame, dict]:
    """Reconstruct the v2 cohort under both history readings and validate both."""
    train = pd.read_csv(TRAIN_V2_CSV, dtype={"msno": str, "is_churn": "int8"})
    train_ids = set(train["msno"])

    # Primary: full history up to the cutoff (best empirical fit).
    last_cutoff, stats = lastday_to_cutoff([TRANSACTIONS_V1_CSV], HISTORY_CUTOFF)
    cand_cutoff = select_candidates(last_cutoff, CANDIDATE_START, CANDIDATE_END)

    # Secondary: the literal v1-style bounded window, for comparison.
    last_window, _ = lastday_to_cutoff([TRANSACTIONS_V1_CSV], HISTORY_CUTOFF,
                                       start=HISTORY_WINDOW_START)
    cand_window = select_candidates(last_window, CANDIDATE_START, CANDIDATE_END)

    primary = _compare(train_ids, set(cand_cutoff["msno"]))
    secondary = _compare(train_ids, set(cand_window["msno"]))

    cohorts = cand_cutoff.merge(train, on="msno", how="inner")
    cohorts["observation_cutoff"] = pd.Timestamp(OBSERVATION_CUTOFF)
    cohorts["cohort"] = COHORT
    cohorts["last_expire"] = pd.to_datetime(cohorts["last_expire"], format="%Y%m%d")
    cohorts = cohorts[["msno", "observation_cutoff", "last_expire", "cohort",
                       "is_churn"]]

    missing = train_ids - set(cand_cutoff["msno"])
    report = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "round": "v2 (March 2017)",
        "source": {
            "train_v2_csv": str(TRAIN_V2_CSV),
            "transactions_csv_v1": str(TRANSACTIONS_V1_CSV),
            "transactions_v2_csv": str(TRANSACTIONS_V2_CSV),
            "labeller": "NONE SHIPPED FOR v2 -- rules inferred from data",
        },
        "windows": {
            "history_rule": "all transaction_date <= cutoff (inferred)",
            "history_cutoff": HISTORY_CUTOFF,
            "candidate_start": CANDIDATE_START,
            "candidate_end": CANDIDATE_END,
            "cohort": COHORT,
            "observation_cutoff": OBSERVATION_CUTOFF.isoformat(),
        },
        "train_v2_rows": int(len(train)),
        "train_v2_unique_msno": int(train["msno"].nunique()),
        "duplicate_msno_count": {
            "train_v2": int(len(train) - train["msno"].nunique()),
            "candidates": int(len(cand_cutoff) - cand_cutoff["msno"].nunique()),
        },
        "reconstruction_full_history_cutoff": primary,
        "reconstruction_literal_v1_window": secondary,
        "churn_rate": {
            "train_v2": round(float(train["is_churn"].mean()), 6),
            "reconstructed_intersection": round(float(cohorts["is_churn"].mean()), 6)
            if len(cohorts) else None,
            "missing_from_reconstruction": round(
                float(train.loc[train["msno"].isin(missing), "is_churn"].mean()), 6)
            if missing else None,
        },
        "date_ranges": {
            "transactions_v1_transaction_date": [stats["transaction_date_min"],
                                                 stats["transaction_date_max"]],
            "reconstructed_last_expire": [last_cutoff["last_expire"].min(),
                                          last_cutoff["last_expire"].max()],
            "candidate_last_expire": [cand_cutoff["last_expire"].min(),
                                      cand_cutoff["last_expire"].max()],
        },
        "sentinel_counts": {
            "sentinel_value": SENTINEL_EXPIRY,
            "rows_in_history": int(stats["sentinel_rows"]),
            "as_last_expire": int((last_cutoff["last_expire"] == SENTINEL_EXPIRY).sum()),
            "in_candidates": int((cand_cutoff["last_expire"] == SENTINEL_EXPIRY).sum()),
        },
        "rows_scanned": int(stats["rows_scanned"]),
        "validation_status": "EXACT_MATCH" if primary["exact_match"] else "MISMATCH",
    }

    if write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        cohorts.to_parquet(COHORTS_PARQUET, index=False)
        VALIDATION_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
        log.info("wrote %s (%s rows)", COHORTS_PARQUET, f"{len(cohorts):,}")

    return cohorts, report


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    for path in (TRAIN_V2_CSV, TRANSACTIONS_V1_CSV):
        if not path.exists():
            parser.error(f"missing required input: {path}")

    _cohorts, report = build_v2(write=not args.dry_run)
    print(json.dumps(report, indent=2))
    return 0 if report["validation_status"] == "EXACT_MATCH" else 1


if __name__ == "__main__":
    raise SystemExit(main())
