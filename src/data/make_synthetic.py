"""
Dev-only helper: generates a schema-accurate synthetic stand-in for the KKBOX
dataset into data/raw/. This is NOT part of the spec's task list -- it exists
so the rest of the pipeline (features, training, API) can be built and tested
in environments without Kaggle access. Swap this out for src/data/ingest.py
output when running against the real competition data; every downstream
module only depends on the column contract in the spec (§2.1), not on this
generator.

Usage:
    python -m src.data.make_synthetic --n-members 2000 --seed 42
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RAW_DIR = Path("data/raw")


def make(n_members: int = 2000, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    msnos = [f"member_{i:06d}" for i in range(n_members)]

    # ---- members.csv ----
    reg_init = pd.to_datetime("2015-01-01") + pd.to_timedelta(
        rng.integers(0, 365 * 4, n_members), unit="D"
    )
    members = pd.DataFrame(
        {
            "msno": msnos,
            "city": rng.integers(1, 22, n_members),
            "bd": np.clip(rng.normal(30, 10, n_members).astype(int), 10, 70),
            "gender": rng.choice(["male", "female", ""], n_members, p=[0.45, 0.45, 0.10]),
            "registered_via": rng.choice([3, 4, 7, 9, 13], n_members),
            "registration_init_time": reg_init.strftime("%Y%m%d").astype(int),
        }
    )

    # ---- transactions.csv (variable count per member) ----
    tx_rows = []
    for msno in msnos:
        n_tx = rng.integers(1, 15)
        last_expire = pd.Timestamp("2017-01-01")
        for _ in range(n_tx):
            tx_date = last_expire - pd.Timedelta(days=int(rng.integers(25, 35)))
            plan_price = int(rng.choice([99, 129, 149, 180, 199]))
            is_cancel = int(rng.random() < 0.05)
            is_auto_renew = int(rng.random() < 0.8)
            expire = tx_date + pd.Timedelta(days=int(rng.choice([30, 90, 365])))
            tx_rows.append(
                {
                    "msno": msno,
                    "payment_method_id": int(rng.integers(1, 41)),
                    "plan_list_price": plan_price,
                    "actual_amount_paid": plan_price - int(rng.integers(0, 10)),
                    "is_auto_renew": is_auto_renew,
                    "transaction_date": int(tx_date.strftime("%Y%m%d")),
                    "membership_expire_date": int(expire.strftime("%Y%m%d")),
                    "is_cancel": is_cancel,
                }
            )
            last_expire = expire
    transactions = pd.DataFrame(tx_rows)

    # ---- user_logs.csv (daily rows per member, last ~120 days) ----
    log_rows = []
    for msno in msnos:
        n_days = rng.integers(10, 120)
        base = pd.Timestamp("2017-01-01")
        engagement_level = rng.random()  # drives listening intensity
        for d in range(n_days):
            date = base - pd.Timedelta(days=int(d))
            secs = max(0, rng.normal(3000 * engagement_level, 800))
            log_rows.append(
                {
                    "msno": msno,
                    "date": int(date.strftime("%Y%m%d")),
                    "num_25": int(rng.poisson(2 * engagement_level)),
                    "num_50": int(rng.poisson(1 * engagement_level)),
                    "num_75": int(rng.poisson(1 * engagement_level)),
                    "num_985": int(rng.poisson(1 * engagement_level)),
                    "num_100": int(rng.poisson(8 * engagement_level)),
                    "num_unq": int(rng.poisson(6 * engagement_level)),
                    "total_secs": round(secs, 2),
                }
            )
    user_logs = pd.DataFrame(log_rows)

    # ---- train.csv (label) ----
    # Correlate churn loosely with auto-renew / recent engagement so the
    # baseline model has real signal to learn, per member's own window.
    last_tx = transactions.sort_values("membership_expire_date").groupby("msno").tail(1)
    last_tx = last_tx.set_index("msno")
    churn_prob = 0.25 - 0.15 * last_tx["is_auto_renew"] + 0.20 * last_tx["is_cancel"]
    churn_prob = churn_prob.clip(0.03, 0.9)
    is_churn = rng.binomial(1, churn_prob.values)
    train = pd.DataFrame({"msno": last_tx.index, "is_churn": is_churn})

    members.to_csv(RAW_DIR / "members.csv", index=False)
    transactions.to_csv(RAW_DIR / "transactions.csv", index=False)
    user_logs.to_csv(RAW_DIR / "user_logs.csv", index=False)
    train.to_csv(RAW_DIR / "train.csv", index=False)

    for name, df in [
        ("train.csv", train),
        ("transactions.csv", transactions),
        ("user_logs.csv", user_logs),
        ("members.csv", members),
    ]:
        print(f"{name:18s} rows={len(df)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-members", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    make(n_members=args.n_members, seed=args.seed)
