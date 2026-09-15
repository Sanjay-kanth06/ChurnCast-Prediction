"""
W2-T1 (corrected): leak-free feature engineering.

Replaces src/data/features.py. That module is left untouched so the current
registered model stays reproducible until this one is validated — run `git init`
before switching over.

WHAT CHANGED AND WHY
--------------------
features.py used GLOBAL_CUTOFF = 2017-03-20, which sits *inside* the label
period: 74.8% of transactions.csv is March 2017, and the March renewals are what
the label is computed from. Measured cost of that leak, on a fixed population of
625,770 members: AUC 0.9513 with the window at 2017-03-20 falling to 0.7512 with
it at 2017-02-01 — roughly 0.20 AUC of the reported score was the model reading
its own answer.

THE CUTOFF DATE
---------------
data/raw/WSDMChurnLabeller.scala is KKBOX's own label generator, and it settles
the date:

    val historyCutoff = "20170131"
    val historyData = data.filter(col("transaction_date") <= lit(historyCutoff))
    val futureData  = data.filter(col("transaction_date") >  lit(historyCutoff))
    val predictionCandidates = userExpire.filter(
        col("last_expire") >= "20170201" and col("last_expire") <= "20170228")

Everything after 2017-01-31 is "future" as far as the label is concerned, so a
February transaction against a February-expiring membership *is* the renewal
decision. 2017-02-28 is therefore still leaky; 2017-01-31 is the correct value
and is the default below.

DATA REQUIREMENTS (read this before trusting any output)
--------------------------------------------------------
This module is correct but cannot produce a usable model from the files
currently in data/raw/:

  * transactions.csv on disk is transactions_v2 (1.43M rows, 75% March 2017) —
    the renewal-window file, not the history file. At FEATURE_CUTOFF only
    14,296 of 970,960 labelled members (1.5%) have any prior transaction, and
    they churn at 63.2% against a true base rate of 9.0%. The competition's
    full-history transactions.csv is required.
  * user_logs.csv on disk is fabricated (make_synthetic_user_logs.py overwrote
    the real file). All six user-log features below will be null or constant
    until the real file is restored.

Both come from the same Kaggle download; src/data/ingest.py already implements
it. Until then, treat every number this module produces as a plumbing test.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("features_v3")

# KKBOX's own historyCutoff — see module docstring.
FEATURE_CUTOFF = pd.Timestamp("2017-01-31")

TXN_WINDOW_DAYS = 90
LOG_WINDOW_DAYS = 30

NUMERIC_FEATURES = [
    # transactions
    "num_transactions",
    "prior_cancellations",
    "num_auto_renewals",
    "avg_discount",
    "plan_price_last",
    "days_since_last_txn",
    # user_logs
    "total_secs_listened",
    "active_days_last_30d",
    "days_since_last_listen",
    "skip_ratio",
    "completion_ratio",
    "avg_daily_songs",
    "engagement_trend",
    # members
    "age",
    "tenure_days",
]
CATEGORICAL_FEATURES = [
    "gender",
    "city",
    "registered_via",
    "payment_method_last",
    # explicit missingness — a member with no history is a real signal, not a gap
    "has_txn_history",
    "has_log_history",
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _to_dt(series: pd.Series) -> pd.Series:
    """KKBOX ships dates as YYYYMMDD integers. Coerce rather than raise — the
    real members.csv has malformed registration dates."""
    return pd.to_datetime(series.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")


TXN_COLS = ["msno", "transaction_date", "plan_list_price", "actual_amount_paid",
            "is_auto_renew", "is_cancel", "payment_method_id"]
MEMBER_COLS = ["msno", "city", "bd", "gender", "registered_via", "registration_init_time"]
LOG_COLS = ["msno", "date", "num_25", "num_50", "num_75", "num_985", "num_100", "total_secs"]

# The real KKBOX transactions.csv is ~21.5M rows; narrow dtypes keep it near 1GB
# instead of several. user_logs.csv is ~30GB and is never loaded whole — see
# build_user_log_features_streaming.
TXN_DTYPES = {"plan_list_price": "int32", "actual_amount_paid": "int32",
              "is_auto_renew": "int8", "is_cancel": "int8", "payment_method_id": "int16"}
LOG_DTYPES = {c: "int32" for c in ["num_25", "num_50", "num_75", "num_985", "num_100"]}
LOG_DTYPES["total_secs"] = "float32"


def load_raw(raw_dir: str | Path = "data/raw") -> dict[str, pd.DataFrame]:
    """Loads everything except user_logs, which is streamed separately because
    the real file does not fit in memory."""
    raw_dir = Path(raw_dir)
    train = pd.read_csv(raw_dir / "train.csv")
    members = pd.read_csv(raw_dir / "members.csv", usecols=MEMBER_COLS)
    transactions = pd.read_csv(raw_dir / "transactions.csv", usecols=TXN_COLS, dtype=TXN_DTYPES)

    transactions["transaction_date"] = _to_dt(transactions["transaction_date"])
    members["registration_init_time"] = _to_dt(members["registration_init_time"])
    return {"train": train, "members": members, "transactions": transactions}


def build_transaction_features(txns: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """One row per member. Only transactions on or before `cutoff` are visible."""
    tx = txns[txns["transaction_date"] <= cutoff].copy()
    if tx.empty:
        return pd.DataFrame(columns=["msno"]).set_index("msno")

    # sort once, so `.last()` means "most recent" rather than "whatever row
    # happened to land last in the group"
    tx = tx.sort_values(["msno", "transaction_date"])

    window = tx[tx["transaction_date"] >= cutoff - pd.Timedelta(TXN_WINDOW_DAYS, unit="D")]

    agg = tx.groupby("msno").agg(
        prior_cancellations=("is_cancel", "sum"),
        num_auto_renewals=("is_auto_renew", "sum"),
        plan_price_last=("plan_list_price", "last"),
        payment_method_last=("payment_method_id", "last"),
        last_txn_date=("transaction_date", "last"),
    )
    agg["num_transactions"] = window.groupby("msno").size().reindex(agg.index).fillna(0)
    agg["days_since_last_txn"] = (cutoff - agg["last_txn_date"]).dt.days

    # vectorised — a groupby.apply here crawls on the full-history file
    list_price = tx["plan_list_price"].replace(0, np.nan)
    tx["_discount"] = (tx["plan_list_price"] - tx["actual_amount_paid"]) / list_price
    agg["avg_discount"] = tx.groupby("msno")["_discount"].mean()

    agg["payment_method_last"] = agg["payment_method_last"].astype("Int64").astype(str)
    return agg.drop(columns=["last_txn_date"])


def build_user_log_features_streaming(
    path: str | Path, cutoff: pd.Timestamp, chunksize: int = 5_000_000
) -> pd.DataFrame:
    """Chunked equivalent of build_user_log_features for the real ~30GB
    user_logs.csv, which pd.read_csv cannot hold.

    Every statistic here is decomposable, so each chunk is reduced to per-member
    partials and folded into a running accumulator. Peak memory is one chunk
    plus the accumulator (~1.2M members), not the file. Ratios and weekly means
    are derived at the end from the accumulated sums, so the result is identical
    to the in-memory path.
    """
    path = Path(path)
    win_start = cutoff - pd.Timedelta(LOG_WINDOW_DAYS, unit="D")
    trend_start = cutoff - pd.Timedelta(28, unit="D")

    hist: pd.DataFrame | None = None   # last_listen over all history <= cutoff
    win: pd.DataFrame | None = None    # sums over the 30-day window
    wk: pd.DataFrame | None = None     # (msno, week_idx) sums for the trend

    def fold(acc: pd.DataFrame | None, part: pd.DataFrame, how: str) -> pd.DataFrame:
        if acc is None:
            return part
        combined = pd.concat([acc, part])
        return combined.groupby(level=list(range(combined.index.nlevels))).agg(how)

    n_rows = 0
    for chunk in pd.read_csv(path, usecols=LOG_COLS, dtype=LOG_DTYPES, chunksize=chunksize):
        chunk["date"] = _to_dt(chunk["date"])
        chunk = chunk[chunk["date"] <= cutoff]
        if chunk.empty:
            continue
        n_rows += len(chunk)

        hist = fold(hist, chunk.groupby("msno")["date"].max().to_frame("last_listen"), "max")

        w = chunk[chunk["date"] >= win_start]
        if not w.empty:
            w = w.assign(_plays=w[["num_25", "num_50", "num_75", "num_985", "num_100"]].sum(axis=1))
            part = w.groupby("msno").agg(
                total_secs_listened=("total_secs", "sum"),
                active_days_last_30d=("date", "nunique"),
                plays=("_plays", "sum"),
                skips=("num_25", "sum"),
                completes=("num_100", "sum"),
            )
            win = fold(win, part, "sum")

        t = chunk[chunk["date"] >= trend_start]
        if not t.empty:
            t = t.assign(week_idx=(cutoff - t["date"]).dt.days // 7)
            part = t.groupby(["msno", "week_idx"]).agg(secs=("total_secs", "sum"),
                                                       n=("total_secs", "size"))
            wk = fold(wk, part, "sum")

    log.info("streamed %s: %d rows on/before %s", path.name, n_rows, cutoff.date())
    if win is None:
        return pd.DataFrame(columns=["msno"]).set_index("msno")

    agg = win.join(hist, how="outer")
    agg["days_since_last_listen"] = (cutoff - agg["last_listen"]).dt.days

    plays = agg["plays"].replace(0, np.nan)
    agg["skip_ratio"] = agg["skips"] / plays
    agg["completion_ratio"] = agg["completes"] / plays
    agg["avg_daily_songs"] = agg["plays"] / agg["active_days_last_30d"].replace(0, np.nan)
    agg["engagement_trend"] = _trend_from_weekly(wk)

    return agg.drop(columns=["plays", "skips", "completes", "last_listen"])


def _trend_from_weekly(wk: pd.DataFrame | None) -> pd.Series:
    """Least-squares slope of weekly mean total_secs, from accumulated sums."""
    if wk is None or wk.empty:
        return pd.Series(dtype=float, name="engagement_trend")
    weekly = wk.reset_index()
    weekly["total_secs"] = weekly["secs"] / weekly["n"]
    weekly["x"] = -weekly["week_idx"]
    g = weekly.groupby("msno")
    xm, ym = g["x"].transform("mean"), g["total_secs"].transform("mean")
    weekly["_num"] = (weekly["x"] - xm) * (weekly["total_secs"] - ym)
    weekly["_den"] = (weekly["x"] - xm) ** 2
    out = weekly.groupby("msno").agg(num=("_num", "sum"), den=("_den", "sum"), k=("x", "size"))
    slope = (out["num"] / out["den"].replace(0, np.nan)).where(out["k"] >= 2, 0.0)
    return slope.fillna(0.0).rename("engagement_trend")


def build_user_log_features(user_logs: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """One row per member. Listening ratios come from the play-count columns
    (num_25 … num_100), which the previous implementation never read."""
    logs = user_logs[user_logs["date"] <= cutoff].copy()
    if logs.empty:
        return pd.DataFrame(columns=["msno"]).set_index("msno")

    play_cols = ["num_25", "num_50", "num_75", "num_985", "num_100"]
    logs["_plays"] = logs[play_cols].sum(axis=1)

    window = logs[logs["date"] >= cutoff - pd.Timedelta(LOG_WINDOW_DAYS, unit="D")]

    agg = window.groupby("msno").agg(
        total_secs_listened=("total_secs", "sum"),
        active_days_last_30d=("date", "nunique"),
        plays=("_plays", "sum"),
        skips=("num_25", "sum"),
        completes=("num_100", "sum"),
    )
    # recency is measured over all history, not just the 30-day window
    agg["days_since_last_listen"] = (
        cutoff - logs.groupby("msno")["date"].max().reindex(agg.index)
    ).dt.days

    plays = agg["plays"].replace(0, np.nan)
    agg["skip_ratio"] = agg["skips"] / plays
    agg["completion_ratio"] = agg["completes"] / plays
    agg["avg_daily_songs"] = agg["plays"] / agg["active_days_last_30d"].replace(0, np.nan)
    agg["engagement_trend"] = _engagement_trend(logs, cutoff)

    return agg.drop(columns=["plays", "skips", "completes"])


def _engagement_trend(logs: pd.DataFrame, cutoff: pd.Timestamp) -> pd.Series:
    """Slope of weekly mean total_secs over the trailing 4 weeks. Positive means
    listening is climbing into the cutoff. 0.0 when fewer than 2 weeks exist."""
    w = logs[logs["date"] >= cutoff - pd.Timedelta(28, unit="D")].copy()
    if w.empty:
        return pd.Series(dtype=float, name="engagement_trend")
    w["week_idx"] = (cutoff - w["date"]).dt.days // 7
    weekly = w.groupby(["msno", "week_idx"])["total_secs"].mean().reset_index()

    # closed-form least-squares slope per member — avoids a per-group polyfit
    weekly["x"] = -weekly["week_idx"]
    g = weekly.groupby("msno")
    xm, ym = g["x"].transform("mean"), g["total_secs"].transform("mean")
    weekly["_num"] = (weekly["x"] - xm) * (weekly["total_secs"] - ym)
    weekly["_den"] = (weekly["x"] - xm) ** 2
    out = weekly.groupby("msno").agg(num=("_num", "sum"), den=("_den", "sum"), k=("x", "size"))
    slope = (out["num"] / out["den"].replace(0, np.nan)).where(out["k"] >= 2, 0.0)
    return slope.fillna(0.0).rename("engagement_trend")


def build_member_features(members: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    m = members.drop_duplicates("msno").set_index("msno")
    age = m["bd"].where(m["bd"].between(10, 80))  # KKBOX `bd` is mostly 0 or junk
    tenure = (cutoff - m["registration_init_time"]).dt.days
    return pd.DataFrame(
        {
            "age": age,
            "tenure_days": tenure.where(tenure >= 0),  # negative = registered after cutoff
            "gender": m["gender"].fillna("unknown").replace("", "unknown"),
            "city": m["city"].astype("Int64").astype(str),
            "registered_via": m["registered_via"].astype("Int64").astype(str),
        }
    )


def build_feature_table(
    raw_dir: str | Path = "data/raw", cutoff: pd.Timestamp = FEATURE_CUTOFF
) -> pd.DataFrame:
    """One row per LABELLED member. Unlike the previous implementation this uses
    a left join, so members with no transaction or log history are retained with
    explicit indicators rather than silently dropped."""
    raw = load_raw(raw_dir)
    labels = raw["train"].drop_duplicates("msno").set_index("msno")["is_churn"]

    txn = build_transaction_features(raw["transactions"], cutoff)
    logs = build_user_log_features_streaming(Path(raw_dir) / "user_logs.csv", cutoff)
    mem = build_member_features(raw["members"], cutoff)

    table = pd.DataFrame(index=labels.index)
    table["has_txn_history"] = table.index.isin(txn.index).astype(int).astype(str)
    table["has_log_history"] = table.index.isin(logs.index).astype(int).astype(str)
    table = table.join(txn, how="left").join(logs, how="left").join(mem, how="left")

    # counts are genuinely zero when there is no history; ratios and recencies
    # stay null so the imputer and the model can tell "none" from "unknown"
    for col in ["num_transactions", "prior_cancellations", "num_auto_renewals",
                "active_days_last_30d", "total_secs_listened", "engagement_trend"]:
        table[col] = table[col].fillna(0)

    table["is_churn"] = labels

    missing = [c for c in ALL_FEATURES if c not in table.columns]
    if missing:
        raise ValueError(f"features declared but not produced: {missing}")

    _log_coverage(table, cutoff)
    return table[ALL_FEATURES + ["is_churn"]]


def _log_coverage(table: pd.DataFrame, cutoff: pd.Timestamp) -> None:
    """The previous implementation dropped 35.6% of labelled members silently,
    and the dropped set churned at 11.32% against 7.71% for those kept. Nothing
    is dropped here, but coverage is logged every run so a collapse in usable
    history can never again hide behind a healthy-looking AUC."""
    log.info("feature cutoff        = %s", cutoff.date())
    log.info("labelled members      = %d  (churn %.4f)", len(table), table["is_churn"].mean())
    for col, label in [("has_txn_history", "transaction"), ("has_log_history", "log")]:
        have = (table[col] == "1")
        log.info(
            "  with %-11s history: %7d (%5.1f%%)  churn %.4f   without: churn %.4f",
            label, have.sum(), 100 * have.mean(),
            table.loc[have, "is_churn"].mean(),
            table.loc[~have, "is_churn"].mean(),
        )
    null_rate = table[ALL_FEATURES].isna().mean()
    worst = null_rate[null_rate > 0.5].sort_values(ascending=False)
    if not worst.empty:
        log.warning("features >50%% null: %s", ", ".join(f"{k} {v:.0%}" for k, v in worst.items()))
    for col in NUMERIC_FEATURES:
        if table[col].nunique(dropna=True) <= 1:
            log.warning("feature %r is constant - it cannot contribute to the model", col)


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                              ("scale", StandardScaler())]), NUMERIC_FEATURES),
            ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                              ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORICAL_FEATURES),
        ]
    )


if __name__ == "__main__":
    ft = build_feature_table()
    print(ft.shape)
    print((ft.isna().mean() * 100).round(2).sort_values(ascending=False).to_string())
