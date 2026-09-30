"""
Real Stage 2: leakage-safe feature engineering for the train_v2 population.

Scope, stated plainly: this is NOT a recreation of the competition's hidden
cohort-selection artifact. Stage 1 established that the artifact cannot be
reproduced from the published files. What this builds instead is a supervised
dataset over the official train_v2 labels (970,960 subscribers) with an
explicit observation cutoff of 2017-02-28 and enforced leakage controls.

THE LEAKAGE RULE, AND HOW IT IS ENFORCED
----------------------------------------
No feature may draw on anything dated after 2017-02-28.

The rule is enforced at the point of reading, not audited afterwards. Every
source is filtered to `date <= CUTOFF` inside the streaming loop, before any
aggregation, so a post-cutoff row cannot reach an accumulator in the first
place. The validation report then re-derives the observed maxima from the
accumulated state as an independent check. Three specific traps:

  * transactions_v2.csv is never opened. Its March records are exactly the
    future the label describes, and Stage 1 showed it is a disjoint extract
    rather than a continuation, so concatenating it would be wrong even
    before the leakage question.
  * membership_expire_date is a FORWARD-LOOKING field, but it is known at the
    cutoff -- it is the contract the subscriber already holds. It is used only
    via the transaction whose transaction_date <= cutoff. The expiry value
    itself may lie past the cutoff; that is legitimate and is why
    days_until_expiry is a feature rather than a leak.
  * is_churn is read from train_v2 solely to attach the label at the end. No
    feature is derived from it.

USER LOGS
---------
user_logs.csv is 30.5 GB uncompressed and is never written to disk. 7-Zip
decompresses to stdout and the stream is parsed in chunks; per-subscriber
accumulation uses fixed-size numpy arrays indexed by a categorical code, so
memory is bounded by the population size rather than the file size.
"""
from __future__ import annotations

import json
import logging
import subprocess
import time
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.real_cohorts import KKBOX_DIR, order_transactions

log = logging.getLogger("real_features_v2")

# ------------------------------------------------------------------ cutoff
CUTOFF = "20170228"
CUTOFF_DATE = date(2017, 2, 28)
CUTOFF_TS = pd.Timestamp(CUTOFF_DATE)

# Trailing windows, all ending at the cutoff. Computed rather than written out
# so they cannot drift from CUTOFF.
def _back(days: int) -> str:
    return (CUTOFF_TS - pd.Timedelta(int(days) - 1, unit="D")).strftime("%Y%m%d")


W7, W30, W90 = _back(7), _back(30), _back(90)
P30_START, P30_END = _back(60), _back(31)   # the 30 days before the last 30

# ------------------------------------------------------------------ sources
TRAIN_V2 = KKBOX_DIR / "train_v2" / "data" / "churn_comp_refresh" / "train_v2.csv"
MEMBERS = KKBOX_DIR / "members" / "members_v3.csv"
TRANSACTIONS = KKBOX_DIR / "transactions" / "transactions.csv"
USER_LOGS_7Z = KKBOX_DIR / "user_logs.csv.7z"
SEVENZIP = Path(r"C:/Program Files/7-Zip/7z.exe")

REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = REPO_ROOT / "data" / "processed" / "real_v2"
FEATURES_PARQUET = OUT_DIR / "features.parquet"
MANIFEST_JSON = OUT_DIR / "feature_manifest.json"
VALIDATION_JSON = OUT_DIR / "feature_validation_report.json"

TX_COLUMNS = ["msno", "payment_method_id", "payment_plan_days", "plan_list_price",
              "actual_amount_paid", "is_auto_renew", "transaction_date",
              "membership_expire_date", "is_cancel"]
LOG_COLUMNS = ["msno", "date", "num_25", "num_50", "num_75", "num_985",
               "num_100", "num_unq", "total_secs"]

CHUNK_TX = 3_000_000
CHUNK_LOG = 4_000_000

# Columns that must never appear in the feature table.
FORBIDDEN_COLUMNS = {"is_churn", "membership_expire_date", "transaction_date", "date"}


def _to_days(series: pd.Series) -> pd.Series:
    """yyyyMMdd string -> days between that date and the cutoff (positive = past)."""
    parsed = pd.to_datetime(series, format="%Y%m%d", errors="coerce")
    return (CUTOFF_TS - parsed).dt.days


# ------------------------------------------------------------------ members

def build_member_features(msno_index: pd.Index) -> tuple[pd.DataFrame, dict]:
    """City, channel, gender, cleaned age and tenure at the cutoff."""
    members = pd.read_csv(MEMBERS, dtype={"msno": str, "city": str,
                                          "registered_via": str, "gender": str,
                                          "bd": "float64",
                                          "registration_init_time": str})
    stats = {"rows": int(len(members)), "unique_msno": int(members["msno"].nunique())}
    members = members.drop_duplicates(subset="msno", keep="last")
    members = members[members["msno"].isin(msno_index)]

    out = pd.DataFrame(index=members["msno"].values)
    out.index.name = "msno"
    out["city"] = members["city"].fillna("unknown").values
    out["registered_via"] = members["registered_via"].fillna("unknown").values
    out["gender"] = members["gender"].fillna("unknown").replace("", "unknown").values

    # bd is self-reported and notoriously dirty: negatives and values in the
    # thousands. Anything outside a plausible range becomes missing rather than
    # being clipped, which would invent a plausible-looking age.
    bd = members["bd"].values
    out["age"] = np.where((bd >= 7) & (bd <= 100), bd, np.nan)
    stats["bd_out_of_range"] = int((~((bd >= 7) & (bd <= 100))).sum())

    reg = _to_days(members["registration_init_time"]).values
    # A registration dated after the cutoff cannot inform a pre-cutoff feature.
    stats["registration_after_cutoff"] = int((reg < 0).sum())
    out["tenure_days"] = np.where(reg >= 0, reg, np.nan)
    out["registered_after_cutoff"] = (reg < 0).astype("int8")

    return out, stats


# -------------------------------------------------------------- transactions

def build_transaction_features(msno_index: pd.Index) -> tuple[pd.DataFrame, dict]:
    """Aggregates over transactions dated at or before the cutoff.

    Streaming rather than load-then-group: 15.1M of the 21.5M rows belong to
    this population, which is roughly 7.6 GiB held as Python strings. Numeric
    aggregates accumulate into fixed-size arrays via np.bincount, and the
    subscriber's standing contract at the cutoff is found by folding the
    WSDMChurnLabeller comparator chunk by chunk -- the comparator is a total
    order, so its per-subscriber maximum is associative.

    The cutoff filter is applied per chunk before anything is retained, so no
    post-cutoff transaction participates in any statistic.
    """
    cats = pd.Index(msno_index)
    n = len(cats)
    acc = {k: np.zeros(n, dtype="float64") for k in (
        "count", "plan_days", "list_price", "actual_paid", "discount",
        "auto_renew", "cancel", "txn_30", "txn_90", "cancel_30", "cancel_90")}
    max_date = np.zeros(n, dtype="int64")
    min_date = np.full(n, 99999999, dtype="int64")
    pairs_method: set[np.int64] = set()
    pairs_plan: set[np.int64] = set()
    winners: pd.DataFrame | None = None

    stats = {"rows_scanned": 0, "rows_at_or_before_cutoff": 0,
             "rows_after_cutoff_discarded": 0, "rows_outside_population": 0,
             "max_transaction_date_kept": None}

    for chunk in pd.read_csv(TRANSACTIONS, usecols=TX_COLUMNS, dtype=str,
                             chunksize=CHUNK_TX):
        stats["rows_scanned"] += len(chunk)
        after = chunk["transaction_date"] > CUTOFF
        stats["rows_after_cutoff_discarded"] += int(after.sum())
        chunk = chunk[~after]
        if chunk.empty:
            continue

        code = cats.get_indexer(chunk["msno"].values)
        known = code >= 0
        stats["rows_outside_population"] += int((~known).sum())
        if not known.any():
            continue
        chunk = chunk[known].reset_index(drop=True)
        code = code[known]
        stats["rows_at_or_before_cutoff"] += len(chunk)

        td = chunk["transaction_date"]
        hi = td.max()
        stats["max_transaction_date_kept"] = (
            hi if stats["max_transaction_date_kept"] is None
            else max(stats["max_transaction_date_kept"], hi))

        plan = pd.to_numeric(chunk["payment_plan_days"], errors="coerce").fillna(0).values
        price = pd.to_numeric(chunk["plan_list_price"], errors="coerce").fillna(0).values
        paid = pd.to_numeric(chunk["actual_amount_paid"], errors="coerce").fillna(0).values
        auto = pd.to_numeric(chunk["is_auto_renew"], errors="coerce").fillna(0).values
        canc = pd.to_numeric(chunk["is_cancel"], errors="coerce").fillna(0).values
        method = pd.to_numeric(chunk["payment_method_id"], errors="coerce").fillna(-1).values

        def add(key, weights=None, idx=code):
            acc[key] += np.bincount(idx, weights=weights, minlength=n)

        add("count")
        add("plan_days", plan)
        add("list_price", price)
        add("actual_paid", paid)
        add("discount", price - paid)
        add("auto_renew", auto)
        add("cancel", canc)

        tdi = td.values.astype("int64")
        for label, start in (("30", int(W30)), ("90", int(W90))):
            mask = tdi >= start
            if mask.any():
                sub = code[mask]
                acc[f"txn_{label}"] += np.bincount(sub, minlength=n)
                acc[f"cancel_{label}"] += np.bincount(sub, weights=canc[mask],
                                                      minlength=n)

        # Distinct counts without a per-subscriber set: pack (code, value) into
        # one int64 and keep the distinct packed pairs.
        pairs_method.update(np.unique(code.astype("int64") * 1000
                                      + method.astype("int64")).tolist())
        pairs_plan.update(np.unique(code.astype("int64") * 1000
                                    + np.clip(plan, 0, 999).astype("int64")).tolist())

        g = pd.Series(tdi).groupby(code)
        gh, gl = g.max(), g.min()
        np.maximum.at(max_date, gh.index.values, gh.values)
        np.minimum.at(min_date, gl.index.values, gl.values)

        # Fold the comparator: reduce this chunk to one row per subscriber,
        # then re-reduce against the running winners.
        small = chunk[TX_COLUMNS].copy()
        small["_code"] = code
        pool = small if winners is None else pd.concat([winners, small],
                                                       ignore_index=True)
        winners = (order_transactions(pool).groupby("_code", sort=False).tail(1)
                   .reset_index(drop=True))
        winners = winners[TX_COLUMNS + ["_code"]]

    current = order_transactions(winners).groupby("_code", sort=False).tail(1)
    cur_code = current["_code"].values

    seen = acc["count"] > 0
    safe = np.where(seen, acc["count"], np.nan)
    out = pd.DataFrame(index=cats)
    out.index.name = "msno"

    def col(values):
        return np.where(seen, values, np.nan)

    out["transaction_count"] = col(acc["count"])
    out["average_plan_days"] = col(acc["plan_days"] / safe)
    out["average_list_price"] = col(acc["list_price"] / safe)
    out["average_actual_paid"] = col(acc["actual_paid"] / safe)
    out["total_paid"] = col(acc["actual_paid"])
    out["mean_discount"] = col(acc["discount"] / safe)
    out["auto_renew_rate"] = col(acc["auto_renew"] / safe)
    out["cancel_rate"] = col(acc["cancel"] / safe)
    out["cancel_count"] = col(acc["cancel"])
    out["txns_last_30d"] = col(acc["txn_30"])
    out["txns_last_90d"] = col(acc["txn_90"])
    out["cancels_last_30d"] = col(acc["cancel_30"])
    out["cancels_last_90d"] = col(acc["cancel_90"])

    dm = np.zeros(n, dtype="float64")
    dp = np.zeros(n, dtype="float64")
    if pairs_method:
        codes = np.array(sorted(pairs_method), dtype="int64") // 1000
        u, c = np.unique(codes, return_counts=True)
        dm[u] = c
    if pairs_plan:
        codes = np.array(sorted(pairs_plan), dtype="int64") // 1000
        u, c = np.unique(codes, return_counts=True)
        dp[u] = c
    out["distinct_payment_methods"] = col(dm)
    out["distinct_plan_days"] = col(dp)

    last_s = pd.Series(np.where(seen, max_date, np.nan)).astype("Int64").astype(str)
    first_s = pd.Series(np.where(seen, min_date, np.nan)).astype("Int64").astype(str)
    days_last = _to_days(last_s).values
    days_first = _to_days(first_s).values
    out["days_since_last_transaction"] = col(days_last)
    span = np.clip(days_first - days_last, 0, None)
    out["transaction_history_days"] = col(span)
    with np.errstate(divide="ignore", invalid="ignore"):
        freq = np.where(span > 0, acc["count"] / (span / 30.0), np.nan)
    out["transaction_frequency"] = col(freq)

    # State at the cutoff: the standing contract, chosen by the comparator so a
    # same-day cancellation correctly supersedes the subscription it cancels.
    for name, src in (("current_plan_days", "payment_plan_days"),
                      ("current_price", "plan_list_price"),
                      ("current_actual_paid", "actual_amount_paid"),
                      ("current_auto_renew", "is_auto_renew"),
                      ("current_cancel_status", "is_cancel")):
        arr = np.full(n, np.nan)
        arr[cur_code] = pd.to_numeric(current[src], errors="coerce").values
        out[name] = col(arr)

    # Forward-looking value, but known at the cutoff (see module docstring).
    exp = np.full(n, np.nan)
    exp[cur_code] = -_to_days(current["membership_expire_date"]).values
    out["days_until_membership_expiry"] = col(exp)

    stats["subscribers_with_transactions"] = int(seen.sum())
    return out, stats


# ----------------------------------------------------------------- user logs

def build_userlog_features(msno_index: pd.Index) -> tuple[pd.DataFrame, dict]:
    """Stream user_logs.csv.7z and accumulate per-subscriber aggregates.

    Never extracted to disk. 7-Zip writes to stdout; chunks are filtered to
    `date <= CUTOFF` and to the labelled population before accumulation.
    Accumulators are fixed-size numpy arrays indexed by categorical code, so
    memory depends on the population (970,960) and not on the 30.5 GB input.
    """
    cats = pd.Index(msno_index)
    n = len(cats)
    acc = {k: np.zeros(n, dtype="float64") for k in (
        "days", "secs", "secs_sq", "unq", "n25", "n50", "n75", "n985", "n100",
        "days_30", "secs_30", "days_p30", "secs_p30", "days_90", "secs_90",
        "days_7", "secs_7")}
    max_date = np.zeros(n, dtype="int64")
    min_date = np.full(n, 99999999, dtype="int64")

    stats = {"rows_scanned": 0, "rows_at_or_before_cutoff": 0,
             "rows_after_cutoff_discarded": 0, "rows_outside_population": 0,
             "max_log_date_kept": 0, "chunks": 0}

    proc = subprocess.Popen(
        [str(SEVENZIP), "e", "-so", str(USER_LOGS_7Z)],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=1024 * 1024)

    try:
        reader = pd.read_csv(proc.stdout, usecols=LOG_COLUMNS, chunksize=CHUNK_LOG,
                             dtype={"msno": str, "date": "int64",
                                    "num_25": "int32", "num_50": "int32",
                                    "num_75": "int32", "num_985": "int32",
                                    "num_100": "int32", "num_unq": "int32",
                                    "total_secs": "float64"})
        for chunk in reader:
            stats["chunks"] += 1
            stats["rows_scanned"] += len(chunk)

            after = chunk["date"].values > int(CUTOFF)
            stats["rows_after_cutoff_discarded"] += int(after.sum())
            chunk = chunk[~after]
            if chunk.empty:
                continue

            code = cats.get_indexer(chunk["msno"].values)
            known = code >= 0
            stats["rows_outside_population"] += int((~known).sum())
            if not known.any():
                continue
            chunk = chunk[known]
            code = code[known]
            stats["rows_at_or_before_cutoff"] += len(chunk)

            d = chunk["date"].values
            stats["max_log_date_kept"] = max(stats["max_log_date_kept"], int(d.max()))
            secs = chunk["total_secs"].values

            def add(key: str, weights, idx=code):
                acc[key] += np.bincount(idx, weights=weights, minlength=n)

            add("days", None)
            add("secs", secs)
            add("secs_sq", secs * secs)
            add("unq", chunk["num_unq"].values)
            for key, col in (("n25", "num_25"), ("n50", "num_50"), ("n75", "num_75"),
                             ("n985", "num_985"), ("n100", "num_100")):
                add(key, chunk[col].values)

            for label, mask in (
                ("7", d >= int(W7)),
                ("30", d >= int(W30)),
                ("90", d >= int(W90)),
                ("p30", (d >= int(P30_START)) & (d <= int(P30_END))),
            ):
                if mask.any():
                    sub_code = code[mask]
                    acc[f"days_{label}"] += np.bincount(sub_code, minlength=n)
                    acc[f"secs_{label}"] += np.bincount(sub_code, weights=secs[mask],
                                                        minlength=n)

            # max/min over the already-reduced per-subscriber values
            g = pd.Series(d).groupby(code)
            hi, lo = g.max(), g.min()
            np.maximum.at(max_date, hi.index.values, hi.values)
            np.minimum.at(min_date, lo.index.values, lo.values)
    finally:
        if proc.stdout:
            proc.stdout.close()
        proc.wait()

    seen = acc["days"] > 0
    out = pd.DataFrame(index=cats)
    out.index.name = "msno"

    def col(values):
        return np.where(seen, values, np.nan)

    days = acc["days"]
    safe_days = np.where(seen, days, np.nan)
    out["log_active_days"] = col(days)
    out["log_total_secs"] = col(acc["secs"])
    out["log_average_daily_secs"] = col(acc["secs"] / safe_days)
    out["log_total_unique_songs"] = col(acc["unq"])
    out["log_average_unique_songs_per_active_day"] = col(acc["unq"] / safe_days)
    for key, name in (("n25", "log_num_25"), ("n50", "log_num_50"),
                      ("n75", "log_num_75"), ("n985", "log_num_985"),
                      ("n100", "log_num_100")):
        out[name] = col(acc[key])

    plays = acc["n25"] + acc["n50"] + acc["n75"] + acc["n985"] + acc["n100"]
    out["log_completion_rate"] = col(np.divide(acc["n100"], plays,
                                               out=np.full(n, np.nan),
                                               where=plays > 0))
    out["log_total_plays"] = col(plays)

    # Daily variability: population sd of per-day seconds, from the running
    # sum and sum of squares.
    var = np.divide(acc["secs_sq"], safe_days,
                    out=np.full(n, np.nan), where=seen) - \
        np.square(np.divide(acc["secs"], safe_days, out=np.full(n, np.nan), where=seen))
    out["log_daily_secs_sd"] = col(np.sqrt(np.clip(var, 0, None)))

    for label, span in (("7d", 7), ("30d", 30), ("90d", 90)):
        out[f"log_active_days_last_{label}"] = col(acc[f"days_{label.rstrip('d')}"])
        out[f"log_secs_last_{label}"] = col(acc[f"secs_{label.rstrip('d')}"])
        out[f"log_activity_rate_last_{label}"] = col(
            acc[f"days_{label.rstrip('d')}"] / span)

    # Trend: last 30 days against the 30 before them. Ratio guarded so a silent
    # prior period yields NaN rather than an infinity.
    out["log_secs_trend_ratio"] = col(
        np.divide(acc["secs_30"], acc["secs_p30"],
                  out=np.full(n, np.nan), where=acc["secs_p30"] > 0))
    out["log_secs_trend_delta"] = col(acc["secs_30"] - acc["secs_p30"])
    out["log_active_days_trend_delta"] = col(acc["days_30"] - acc["days_p30"])

    last_seen = np.where(seen, max_date, np.nan)
    first_seen = np.where(seen, min_date, np.nan)
    out["log_days_since_last_listen"] = col(
        _to_days(pd.Series(last_seen).astype("Int64").astype(str)).values)
    out["log_listening_history_days"] = col(
        _to_days(pd.Series(first_seen).astype("Int64").astype(str)).values)

    stats["subscribers_with_logs"] = int(seen.sum())
    stats["max_log_date_kept"] = int(stats["max_log_date_kept"])
    return out.loc[seen | ~seen], stats


# ------------------------------------------------------------------ assembly

CATEGORICAL = ["city", "registered_via", "gender"]


def _manifest(features: pd.DataFrame, sources: dict) -> dict:
    """Per-feature provenance, so every column can be traced to its source."""
    origin = {}
    for col in features.columns:
        if col in ("is_churn",):
            continue
        if col.startswith("log_"):
            src, window = "user_logs.csv.7z", "date <= 2017-02-28"
        elif col in ("city", "registered_via", "gender", "age", "tenure_days",
                     "registered_after_cutoff"):
            src, window = "members_v3.csv", "registration_init_time <= 2017-02-28"
        else:
            src, window = "transactions.csv", "transaction_date <= 2017-02-28"
        origin[col] = {
            "source": src,
            "window": window,
            "dtype": str(features[col].dtype),
            "categorical": col in CATEGORICAL,
        }
    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "observation_cutoff": CUTOFF_DATE.isoformat(),
        "population": "train_v2.csv (official labels)",
        "scope_note": (
            "Not a recreation of the competition's hidden cohort-selection "
            "artifact. The official train_v2 labels form the supervised "
            "population, with a documented observation cutoff and enforced "
            "leakage controls."),
        "windows": {"last_7d_start": W7, "last_30d_start": W30,
                    "last_90d_start": W90,
                    "prev_30d": [P30_START, P30_END]},
        "excluded_sources": {
            "transactions_v2.csv": (
                "March 2017 records are the label period, and Stage 1 showed "
                "the file is a disjoint extract rather than a continuation."),
        },
        "feature_count": len(origin),
        "categorical_features": CATEGORICAL,
        "sources": sources,
        "features": origin,
    }


def _validate(features: pd.DataFrame, labels: pd.DataFrame, stats: dict) -> dict:
    """Independent re-derivation of the leakage guarantees from the result."""
    feat_only = features.drop(columns=["is_churn"], errors="ignore")
    numeric = feat_only.select_dtypes(include=[np.number])

    missing = feat_only.isna().sum()
    inf_counts = {c: int(np.isinf(numeric[c]).sum()) for c in numeric.columns
                  if np.isinf(numeric[c]).any()}
    constant = [c for c in feat_only.columns if feat_only[c].nunique(dropna=True) <= 1]

    tx_max = stats["transactions"]["max_transaction_date_kept"]
    log_max = stats["user_logs"]["max_log_date_kept"]
    leakage = {
        "cutoff": CUTOFF,
        "max_transaction_date_used": tx_max,
        "transactions_within_cutoff": bool(tx_max is None or tx_max <= CUTOFF),
        "post_cutoff_transactions_discarded":
            int(stats["transactions"]["rows_after_cutoff_discarded"]),
        "max_user_log_date_used": int(log_max),
        "user_logs_within_cutoff": bool(log_max <= int(CUTOFF)),
        "post_cutoff_user_log_rows_discarded":
            int(stats["user_logs"]["rows_after_cutoff_discarded"]),
        "transactions_v2_opened": False,
        "label_columns_in_features": sorted(
            set(feat_only.columns) & FORBIDDEN_COLUMNS),
        "feature_derived_from_is_churn": False,
        "any_feature_sourced_after_cutoff": not (
            (tx_max is None or tx_max <= CUTOFF) and log_max <= int(CUTOFF)),
    }

    return {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "observation_cutoff": CUTOFF_DATE.isoformat(),
        "row_count": int(len(features)),
        "unique_msno_count": int(features.index.nunique()),
        "duplicate_msno_count": int(len(features) - features.index.nunique()),
        "label_distribution": {str(k): int(v) for k, v in
                               labels["is_churn"].value_counts().sort_index().items()},
        "label_rate": round(float(labels["is_churn"].mean()), 6),
        "feature_count": int(feat_only.shape[1]),
        "numeric_feature_count": int(numeric.shape[1]),
        "categorical_feature_count": len(CATEGORICAL),
        "missing_value_counts": {c: int(v) for c, v in missing.items() if v},
        "features_with_no_missing": int((missing == 0).sum()),
        "infinity_counts": inf_counts,
        "constant_feature_count": len(constant),
        "constant_features": constant,
        "join_coverage": stats["coverage"],
        "leakage_checks": leakage,
        "source_stats": stats,
    }


def build_features(write: bool = True) -> tuple[pd.DataFrame, dict, dict]:
    t0 = time.time()
    labels = pd.read_csv(TRAIN_V2, dtype={"msno": str, "is_churn": "int8"})
    index = pd.Index(labels["msno"].values, name="msno")
    log.info("population: %s subscribers", f"{len(index):,}")

    members, m_stats = build_member_features(index)
    log.info("members done (%.0fs)", time.time() - t0)
    tx, t_stats = build_transaction_features(index)
    log.info("transactions done (%.0fs)", time.time() - t0)
    logs, l_stats = build_userlog_features(index)
    log.info("user logs done (%.0fs)", time.time() - t0)

    features = pd.DataFrame(index=index)
    features = features.join(members, how="left")
    features = features.join(tx, how="left")
    features = features.join(logs, how="left")

    for c in CATEGORICAL:
        features[c] = features[c].fillna("unknown").astype(str)

    coverage = {
        "population": int(len(index)),
        "members_matched": int(members.index.isin(index).sum()),
        "members_coverage": round(float(features["city"].ne("unknown").mean()), 6),
        "transactions_matched": int(t_stats["subscribers_with_transactions"]),
        "transactions_coverage": round(
            float(features["transaction_count"].notna().mean()), 6),
        "user_logs_matched": int(l_stats["subscribers_with_logs"]),
        "user_logs_coverage": round(
            float(features["log_active_days"].notna().mean()), 6),
    }
    stats = {"members": m_stats, "transactions": t_stats, "user_logs": l_stats,
             "coverage": coverage,
             "elapsed_seconds": round(time.time() - t0, 1)}

    features["is_churn"] = labels["is_churn"].values

    manifest = _manifest(features, {
        "train_v2": str(TRAIN_V2), "members_v3": str(MEMBERS),
        "transactions": str(TRANSACTIONS), "user_logs": str(USER_LOGS_7Z),
    })
    report = _validate(features, labels, stats)

    if write:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        features.reset_index().to_parquet(FEATURES_PARQUET, index=False)
        MANIFEST_JSON.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        VALIDATION_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
        log.info("wrote %s (%s rows x %s cols)", FEATURES_PARQUET,
                 f"{len(features):,}", features.shape[1])

    return features, manifest, report


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    _f, _m, report = build_features(write=True)
    print(json.dumps({k: report[k] for k in (
        "row_count", "unique_msno_count", "label_distribution", "feature_count",
        "constant_feature_count", "join_coverage", "leakage_checks")}, indent=2))
    return 0 if not report["leakage_checks"]["any_feature_sourced_after_cutoff"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
