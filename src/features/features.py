"""
Leakage-aware feature engineering for the ChurnCast prototype.

THE LEAKAGE RULE
----------------
Every subscriber has their own OBSERVATION CUTOFF -- the membership expiry date
that the churn question is asked about. A feature may only read records dated on
or before that subscriber's cutoff. The renewal transaction that determines the
label falls strictly after it, so a correct implementation can never see it.

Two categories of field are deliberately excluded from the feature space even
though they exist in the raw data:

  * membership_expire_date -- the label is defined against it, so any feature
    derived from it (days until expiry, expiry month, and so on) is a restatement
    of the question rather than evidence about the answer.
  * anything from a transaction or log row dated after the cutoff.

Both exclusions are enforced in code, not by convention, and tested in
tests/test_features.py.
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

from src import config

log = logging.getLogger("features")

NUMERIC_FEATURES = [
    # recency
    "recency_days",
    "days_since_last_listen",
    # transaction frequency
    "txns_last_30d",
    "txns_last_90d",
    "transaction_frequency",
    "prior_cancellations",
    # listening frequency / volume
    "active_days_last_30d",
    "active_days_last_90d",
    "total_secs_last_30d",
    "total_secs_last_90d",
    "avg_daily_secs",
    "unique_songs_last_30d",
    "unique_songs_last_90d",
    # behavioural quality
    "skip_ratio",
    "completion_ratio",
    "engagement_trend",
    "recent_activity_ratio",
    # subscription economics
    "plan_price",
    "payment_amount",
    "payment_consistency",
    # lifecycle
    "tenure_days",
    "age",
]
CATEGORICAL_FEATURES = [
    "is_auto_renew",
    "gender",
    "city",
    "registered_via",
    "payment_method_last",
]
FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Fields that must never become features: the label is defined against them.
FORBIDDEN_SOURCE_COLUMNS = {"membership_expire_date", "is_churn"}


def _to_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s.astype("Int64").astype(str), format="%Y%m%d", errors="coerce")


def load_raw(raw_dir: str | Path | None = None) -> dict[str, pd.DataFrame]:
    raw_dir = Path(raw_dir) if raw_dir else config.RAW_DIR
    train = pd.read_csv(raw_dir / "train.csv")
    members = pd.read_csv(raw_dir / "members.csv")
    transactions = pd.read_csv(raw_dir / "transactions.csv")
    user_logs = pd.read_csv(raw_dir / "user_logs.csv")
    cohorts = pd.read_csv(raw_dir / "cohorts.csv")

    transactions["transaction_date"] = _to_dt(transactions["transaction_date"])
    user_logs["date"] = _to_dt(user_logs["date"])
    members["registration_init_time"] = _to_dt(members["registration_init_time"])
    cohorts["observation_cutoff"] = _to_dt(cohorts["observation_cutoff"])
    return {"train": train, "members": members, "transactions": transactions,
            "user_logs": user_logs, "cohorts": cohorts}


def _transaction_features(tx: pd.DataFrame, cutoff: pd.DataFrame) -> pd.DataFrame:
    """Per-subscriber transaction aggregates, windowed against each subscriber's
    own cutoff. membership_expire_date is dropped up front so it cannot leak."""
    tx = tx.drop(columns=[c for c in FORBIDDEN_SOURCE_COLUMNS if c in tx.columns])
    tx = tx.merge(cutoff, on="msno", how="inner")
    tx = tx[tx["transaction_date"] <= tx["observation_cutoff"]]        # LEAKAGE GUARD
    if tx.empty:
        return pd.DataFrame(index=pd.Index([], name="msno"))

    tx = tx.sort_values(["msno", "transaction_date"])
    tx["age_days"] = (tx["observation_cutoff"] - tx["transaction_date"]).dt.days

    last = tx.groupby("msno").tail(1).set_index("msno")
    out = pd.DataFrame(index=last.index)
    out["recency_days"] = last["age_days"]
    out["plan_price"] = last["plan_list_price"]
    out["payment_amount"] = last["actual_amount_paid"]
    out["is_auto_renew"] = last["is_auto_renew"].astype("Int64").astype(str)
    out["payment_method_last"] = last["payment_method_id"].astype("Int64").astype(str)

    out["txns_last_30d"] = tx[tx.age_days <= 30].groupby("msno").size().reindex(out.index).fillna(0)
    out["txns_last_90d"] = tx[tx.age_days <= 90].groupby("msno").size().reindex(out.index).fillna(0)
    out["prior_cancellations"] = tx.groupby("msno")["is_cancel"].sum().reindex(out.index).fillna(0)

    # transactions per 30 days of observed history
    span = (tx.groupby("msno")["age_days"].max() - tx.groupby("msno")["age_days"].min()).clip(lower=1)
    out["transaction_frequency"] = (tx.groupby("msno").size() / span * 30).reindex(out.index)

    # how consistently the subscriber pays the list price (1.0 = always full price)
    paid_ratio = (tx["actual_amount_paid"] / tx["plan_list_price"].replace(0, np.nan)).clip(0, 2)
    out["payment_consistency"] = paid_ratio.groupby(tx["msno"]).mean().reindex(out.index)
    return out


def _log_features(logs: pd.DataFrame, cutoff: pd.DataFrame) -> pd.DataFrame:
    """Per-subscriber listening aggregates over 30/90-day windows ending at each
    subscriber's own cutoff."""
    logs = logs.merge(cutoff, on="msno", how="inner")
    logs = logs[logs["date"] <= logs["observation_cutoff"]]            # LEAKAGE GUARD
    if logs.empty:
        return pd.DataFrame(index=pd.Index([], name="msno"))

    logs["age_days"] = (logs["observation_cutoff"] - logs["date"]).dt.days
    play_cols = ["num_25", "num_50", "num_75", "num_985", "num_100"]
    logs["plays"] = logs[play_cols].sum(axis=1)

    w30 = logs[logs.age_days <= 30]
    w90 = logs[logs.age_days <= 90]

    idx = logs["msno"].drop_duplicates()
    out = pd.DataFrame(index=pd.Index(idx, name="msno"))

    out["days_since_last_listen"] = logs.groupby("msno")["age_days"].min().reindex(out.index)
    out["active_days_last_30d"] = w30.groupby("msno")["date"].nunique().reindex(out.index).fillna(0)
    out["active_days_last_90d"] = w90.groupby("msno")["date"].nunique().reindex(out.index).fillna(0)
    out["total_secs_last_30d"] = w30.groupby("msno")["total_secs"].sum().reindex(out.index).fillna(0)
    out["total_secs_last_90d"] = w90.groupby("msno")["total_secs"].sum().reindex(out.index).fillna(0)
    out["unique_songs_last_30d"] = w30.groupby("msno")["num_unq"].sum().reindex(out.index).fillna(0)
    out["unique_songs_last_90d"] = w90.groupby("msno")["num_unq"].sum().reindex(out.index).fillna(0)
    out["avg_daily_secs"] = out["total_secs_last_30d"] / out["active_days_last_30d"].replace(0, np.nan)

    plays90 = w90.groupby("msno")["plays"].sum().reindex(out.index)
    out["skip_ratio"] = (w90.groupby("msno")["num_25"].sum().reindex(out.index)
                         / plays90.replace(0, np.nan))
    out["completion_ratio"] = (w90.groupby("msno")["num_100"].sum().reindex(out.index)
                               / plays90.replace(0, np.nan))

    # is recent listening above or below the subscriber's own 90-day baseline?
    base = (out["total_secs_last_90d"] / 90).replace(0, np.nan)
    out["recent_activity_ratio"] = (out["total_secs_last_30d"] / 30) / base

    out["engagement_trend"] = _engagement_trend(logs, out.index)
    return out


def _engagement_trend(logs: pd.DataFrame, index: pd.Index) -> pd.Series:
    """Least-squares slope of weekly mean listening seconds over the trailing
    four weeks. Positive means engagement is rising into the cutoff."""
    w = logs[logs.age_days <= config.TREND_WINDOW_DAYS].copy()
    if w.empty:
        return pd.Series(np.nan, index=index, name="engagement_trend")
    w["week"] = -(w["age_days"] // 7)                   # -3 oldest .. 0 most recent
    weekly = w.groupby(["msno", "week"])["total_secs"].mean().reset_index()

    g = weekly.groupby("msno")
    xm = g["week"].transform("mean")
    ym = g["total_secs"].transform("mean")
    weekly["_num"] = (weekly["week"] - xm) * (weekly["total_secs"] - ym)
    weekly["_den"] = (weekly["week"] - xm) ** 2
    agg = weekly.groupby("msno").agg(num=("_num", "sum"), den=("_den", "sum"), k=("week", "size"))
    slope = (agg["num"] / agg["den"].replace(0, np.nan)).where(agg["k"] >= 2, 0.0)
    return slope.reindex(index).rename("engagement_trend")


def _member_features(members: pd.DataFrame, cutoff: pd.DataFrame) -> pd.DataFrame:
    m = members.drop_duplicates("msno").merge(cutoff, on="msno", how="inner").set_index("msno")
    tenure = (m["observation_cutoff"] - m["registration_init_time"]).dt.days
    return pd.DataFrame({
        "age": m["bd"].where(m["bd"].between(13, 75)),          # 0 and outliers -> missing
        "tenure_days": tenure.where(tenure >= 0),
        "gender": m["gender"].fillna("unknown").replace("", "unknown"),
        "city": m["city"].astype("Int64").astype(str),
        "registered_via": m["registered_via"].astype("Int64").astype(str),
    })


def build_feature_table(raw_dir: str | Path | None = None, save: bool = True) -> pd.DataFrame:
    """One row per labelled subscriber: FEATURE_COLUMNS + is_churn + cohort_month.

    Subscribers lacking transaction or listening history are retained rather than
    dropped, so the modelled population matches the labelled population.
    """
    raw = load_raw(raw_dir)
    cutoff = raw["cohorts"][["msno", "observation_cutoff"]]
    labels = raw["train"].drop_duplicates("msno").set_index("msno")["is_churn"]

    txf = _transaction_features(raw["transactions"], cutoff)
    logf = _log_features(raw["user_logs"], cutoff)
    memf = _member_features(raw["members"], cutoff)

    table = pd.DataFrame(index=labels.index)
    table = table.join(txf, how="left").join(logf, how="left").join(memf, how="left")

    # counts are genuinely zero without history; ratios/recencies stay missing so
    # the imputer can distinguish "none observed" from "unknown"
    for col in ["txns_last_30d", "txns_last_90d", "prior_cancellations",
                "active_days_last_30d", "active_days_last_90d",
                "total_secs_last_30d", "total_secs_last_90d",
                "unique_songs_last_30d", "unique_songs_last_90d"]:
        table[col] = table[col].fillna(0)

    table["is_churn"] = labels
    table = table.join(raw["cohorts"].set_index("msno")["cohort_month"], how="left")

    missing = [c for c in FEATURE_COLUMNS if c not in table.columns]
    if missing:
        raise ValueError(f"declared features not produced: {missing}")

    table = table[FEATURE_COLUMNS + ["is_churn", "cohort_month"]]
    # same normalisation the loader applies, so training and serving see
    # identical categorical values regardless of which path produced the frame
    table = normalise_categoricals(table)
    _log_diagnostics(table)

    if save:
        config.ensure_dirs()
        table.to_csv(config.FEATURE_TABLE_CSV)
        log.info("feature table written to %s", config.FEATURE_TABLE_CSV)
    return table


def normalise_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Forces the categorical columns into one canonical string form.

    This exists because of a train/serve skew that is easy to introduce and
    silent when it happens: the builder produces `is_auto_renew == "1"`, but a
    CSV round-trip reads that column back as float 1.0. `str(1.0)` is "1.0",
    which the fitted OneHotEncoder has never seen, so `handle_unknown="ignore"`
    would encode it as all-zeros and quietly discard the feature at serving
    time. Applying the same normalisation on build and on load keeps the two
    paths byte-identical.
    """
    out = df.copy()
    for col in CATEGORICAL_FEATURES:
        if col not in out.columns:
            continue
        s = out[col]
        numeric = pd.to_numeric(s, errors="coerce")
        # numeric-valued codes go through Int64 so 1.0 and "1" both become "1"
        if numeric.notna().any():
            out[col] = numeric.astype("Int64").astype(str).replace("<NA>", "unknown")
        else:
            out[col] = s.fillna("unknown").astype(str).replace("", "unknown")
    return out


def load_feature_table() -> pd.DataFrame:
    """Reads the persisted feature table (used by the API for msno lookup)."""
    if not config.FEATURE_TABLE_CSV.exists():
        raise FileNotFoundError(
            f"{config.FEATURE_TABLE_CSV} not found. Run scripts/run_pipeline.py first."
        )
    df = pd.read_csv(config.FEATURE_TABLE_CSV, index_col="msno")
    return normalise_categoricals(df)


def _log_diagnostics(table: pd.DataFrame) -> None:
    log.info("feature table: %d subscribers, churn rate %.4f",
             len(table), table["is_churn"].mean())
    nulls = table[FEATURE_COLUMNS].isna().mean()
    for col, rate in nulls[nulls > 0.30].sort_values(ascending=False).items():
        log.warning("feature %r is %.0f%% null", col, 100 * rate)
    for col in NUMERIC_FEATURES:
        if table[col].nunique(dropna=True) <= 1:
            log.warning("feature %r is constant - it cannot contribute", col)


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), NUMERIC_FEATURES),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]),
         CATEGORICAL_FEATURES),
    ])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    t = build_feature_table()
    print(t.shape)
    print(t[NUMERIC_FEATURES].describe().T[["count", "mean", "std", "min", "max"]].round(3))
