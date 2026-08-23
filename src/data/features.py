"""
W2-T1: Feature engineering pipeline.

Implements src/data/feature_spec.yaml as a leakage-safe, one-row-per-member
feature table, plus a scikit-learn preprocessing Pipeline (imputation +
encoding) that src/models/train.py fits on top of it.

Leakage rule (§2.2 / §2.3): every feature for a member is computed using only
transaction/log rows dated on or before a shared OBSERVATION CUTOFF date.

v2 CHANGE: the first version of this module used each member's own *latest*
membership_expire_date as their anchor. That was circular: a member who kept
renewing all year ended up with a late anchor date (lots of history to look
back on), while a member who churned early had an anchor date sitting right
at their last-ever transaction (because nothing came after it). The model
picked up on that artifact -- "how far out is this member's own story-end" --
rather than on genuine behavioral churn signal, which is why every /predict
call above came back at ~97-99% regardless of the input features.

The fix: GLOBAL_CUTOFF below is one fixed calendar date applied identically
to every member. Features only ever see transaction/log rows on or before
that date; the label (already provided by KKBOX in train.csv) reflects
whether the member churned after it. This keeps the same spirit as the
per-member language in §2.2 (never use data from after the point we're
predicting from) while removing the "story stopped early" circularity that a
per-member "latest ever" anchor introduced. This is a deliberate, documented
deviation from a strict per-member window -- flagging per §8.4 rather than
re-introducing the leaky version to match the literal spec wording.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "recency_days",
    "days_since_last_listen",
    "txns_last_90d",
    "active_days_last_30d",
    "engagement_trend",
    "plan_price",
    "prior_cancellations",
    "age",
    "tenure_days",
]
CATEGORICAL_FEATURES = ["gender", "registered_via", "is_auto_renew"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def _to_dt(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series.astype(int).astype(str), format="%Y%m%d")


def load_raw(raw_dir: str | Path = "data/raw") -> dict[str, pd.DataFrame]:
    raw_dir = Path(raw_dir)
    train = pd.read_csv(raw_dir / "train.csv")
    members = pd.read_csv(raw_dir / "members.csv")
    transactions = pd.read_csv(raw_dir / "transactions.csv")
    user_logs = pd.read_csv(raw_dir / "user_logs.csv")

    transactions["transaction_date"] = _to_dt(transactions["transaction_date"])
    transactions["membership_expire_date"] = _to_dt(transactions["membership_expire_date"])
    user_logs["date"] = _to_dt(user_logs["date"])
    members["registration_init_time"] = _to_dt(members["registration_init_time"])

    return {
        "train": train,
        "members": members,
        "transactions": transactions,
        "user_logs": user_logs,
    }


# Fixed observation cutoff shared by every member (see module docstring for
# why this replaced a per-member "latest transaction ever" anchor). Chosen to
# sit before the Feb/Mar 2017 window KKBOX's train_v2 labels are defined
# against, so features never peek at the renew-or-not decision itself.
GLOBAL_CUTOFF = pd.Timestamp("2017-03-20")


def _anchor_dates(transactions: pd.DataFrame) -> pd.Series:
    """Same fixed GLOBAL_CUTOFF for every member with at least one
    transaction on or before it. Returned as a msno-indexed Series (rather
    than a bare constant) purely so downstream merge/join code is unchanged
    from the per-member version."""
    msnos_with_history = transactions.loc[
        transactions["transaction_date"] <= GLOBAL_CUTOFF, "msno"
    ].unique()
    return pd.Series(GLOBAL_CUTOFF, index=pd.Index(msnos_with_history, name="msno"), name="anchor_date")


def _recency_and_plan_features(transactions: pd.DataFrame, anchor: pd.Series) -> pd.DataFrame:
    tx = transactions.merge(anchor, on="msno")
    tx = tx[tx["transaction_date"] <= tx["anchor_date"]]  # leakage guard

    last_tx = tx.sort_values("transaction_date").groupby("msno").tail(1).set_index("msno")
    recency_days = (last_tx["anchor_date"] - last_tx["transaction_date"]).dt.days.rename(
        "recency_days"
    )
    plan_price = last_tx["plan_list_price"].rename("plan_price")
    is_auto_renew = last_tx["is_auto_renew"].rename("is_auto_renew")

    window_90 = tx[tx["transaction_date"] >= tx["anchor_date"] - pd.Timedelta(days=90)]
    txns_last_90d = window_90.groupby("msno").size().rename("txns_last_90d")

    prior = tx.merge(
        last_tx["transaction_date"].rename("last_tx_date"), on="msno", how="left"
    )
    prior_cancel_mask = (prior["transaction_date"] < prior["last_tx_date"]) & (
        prior["is_cancel"] == 1
    )
    prior_cancellations = (
        prior[prior_cancel_mask].groupby("msno").size().rename("prior_cancellations")
    )

    out = pd.concat([recency_days, plan_price, is_auto_renew, txns_last_90d], axis=1)
    out["prior_cancellations"] = prior_cancellations
    return out


def _engagement_features(user_logs: pd.DataFrame, anchor: pd.Series) -> pd.DataFrame:
    logs = user_logs.merge(anchor, on="msno")
    logs = logs[logs["date"] <= logs["anchor_date"]]  # leakage guard

    last_log = logs.sort_values("date").groupby("msno").tail(1).set_index("msno")
    days_since_last_listen = (last_log["anchor_date"] - last_log["date"]).dt.days.rename(
        "days_since_last_listen"
    )

    window_30 = logs[logs["date"] >= logs["anchor_date"] - pd.Timedelta(days=30)]
    active_days_last_30d = (
        window_30.groupby("msno")["date"].nunique().rename("active_days_last_30d")
    )

    window_28 = logs[logs["date"] >= logs["anchor_date"] - pd.Timedelta(days=28)].copy()
    window_28["week_idx"] = (
        (window_28["anchor_date"] - window_28["date"]).dt.days // 7
    )
    weekly = window_28.groupby(["msno", "week_idx"])["total_secs"].mean().reset_index()

    def slope(group: pd.DataFrame) -> float:
        if group["week_idx"].nunique() < 2:
            return 0.0
        x = -group["week_idx"].to_numpy()
        y = group["total_secs"].to_numpy()
        return float(np.polyfit(x, y, 1)[0])

    engagement_trend = (
        weekly.groupby("msno")[["week_idx", "total_secs"]]
        .apply(slope, include_groups=False)
        .rename("engagement_trend")
    )

    out = pd.concat([days_since_last_listen, active_days_last_30d], axis=1)
    out["engagement_trend"] = engagement_trend
    return out


def _demographic_features(members: pd.DataFrame, anchor: pd.Series) -> pd.DataFrame:
    df = members.merge(anchor, on="msno").set_index("msno")
    age = df["bd"].where(df["bd"].between(10, 100))
    gender = df["gender"].fillna("unknown").replace("", "unknown")
    tenure_days = (df["anchor_date"] - df["registration_init_time"]).dt.days
    return pd.DataFrame(
        {
            "age": age,
            "gender": gender,
            "registered_via": df["registered_via"].astype(str),
            "tenure_days": tenure_days,
        }
    )


def build_feature_table(raw_dir: str | Path = "data/raw") -> pd.DataFrame:
    """Returns one row per member: ALL_FEATURES + is_churn, indexed by msno."""
    raw = load_raw(raw_dir)
    anchor = _anchor_dates(raw["transactions"])

    recency_plan = _recency_and_plan_features(raw["transactions"], anchor)
    engagement = _engagement_features(raw["user_logs"], anchor)
    demo = _demographic_features(raw["members"], anchor)

    table = recency_plan.join(engagement, how="left").join(demo, how="left")
    table["active_days_last_30d"] = table["active_days_last_30d"].fillna(0)
    table["engagement_trend"] = table["engagement_trend"].fillna(0.0)
    table["prior_cancellations"] = table["prior_cancellations"].fillna(0)
    table["txns_last_90d"] = table["txns_last_90d"].fillna(0)
    table["is_auto_renew"] = table["is_auto_renew"].astype("Int64").astype(str)

    labels = raw["train"].set_index("msno")["is_churn"]
    table = table.join(labels, how="inner")
    table = table.join(anchor, how="left")

    missing_features = [c for c in ALL_FEATURES if c not in table.columns]
    if missing_features:
        raise ValueError(f"feature_spec.yaml features missing from output: {missing_features}")

    return table[ALL_FEATURES + ["is_churn", "anchor_date"]]


class FeatureTableBuilder(BaseEstimator, TransformerMixin):
    def __init__(self, raw_dir: str | Path = "data/raw"):
        self.raw_dir = raw_dir

    def fit(self, X=None, y=None):
        return self

    def transform(self, X=None) -> pd.DataFrame:
        return build_feature_table(self.raw_dir)


def build_preprocessor() -> ColumnTransformer:
    numeric_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC_FEATURES),
            ("cat", categorical_pipe, CATEGORICAL_FEATURES),
        ]
    )


if __name__ == "__main__":
    ft = build_feature_table()
    print(ft.shape)
    print(ft.isna().mean().sort_values(ascending=False))
    print(ft["is_churn"].value_counts(normalize=True))