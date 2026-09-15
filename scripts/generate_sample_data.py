"""
Deterministic synthetic dataset generator for the ChurnCast prototype.

This produces KKBOX-LIKE data. It is NOT the official KKBOX dataset, and metrics
obtained from it must not be read as KKBOX benchmark results.

Design notes
------------
Each subscriber is assigned a membership expiry date in one of several monthly
cohorts. That expiry date is the subscriber's OBSERVATION CUTOFF: features may
only be built from records dated on or before it. Churn is then decided in the
30 days that follow -- a renewal transaction inside that window means the
subscriber stayed, its absence means they churned.

Two consequences matter for the rest of the pipeline:

  * The renewal transaction that defines the label is written AFTER the cutoff,
    so a leakage-aware feature builder never sees it and a careless one does.
    This makes the leakage guard demonstrable rather than theoretical.
  * Because cohorts are spread across months, the cohort month gives a genuine
    temporal axis to split train / validation / test on.

Churn probability is driven by latent subscriber traits (engagement level,
engagement trend, auto-renew propensity, price, tenure, cancellation history)
combined in a logistic form with noise, so the problem is learnable but not
trivially so.

Usage:
    python scripts/generate_sample_data.py [--n-subscribers 7500] [--seed 42]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402

PAYMENT_METHODS = [36, 38, 39, 40, 41, 32, 34, 37]
PLAN_DAYS = [30, 30, 30, 30, 90, 180, 410]
PLAN_PRICES = {30: [99, 129, 149, 180], 90: [298, 350], 180: [588], 410: [1788]}
CITIES = [1, 3, 4, 5, 6, 8, 12, 13, 14, 15, 18, 22]
REGISTERED_VIA = [3, 4, 7, 9, 13]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate(n_subscribers: int = config.N_SUBSCRIBERS, seed: int = config.SEED) -> dict:
    rng = np.random.default_rng(seed)
    config.ensure_dirs()

    msnos = np.array([config.msno(i + 1) for i in range(n_subscribers)])

    # ---------------------------------------------------------------- cohorts
    # Expiry day is spread within the cohort month so that cutoffs vary.
    cohort_idx = rng.integers(0, len(config.COHORT_MONTHS), n_subscribers)
    cohort_month = np.array(config.COHORT_MONTHS)[cohort_idx]
    month_start = pd.to_datetime(pd.Series(cohort_month) + "-01")
    day_offset = rng.integers(0, 28, n_subscribers)
    expire_date = month_start + pd.to_timedelta(day_offset, unit="D")

    # ---------------------------------------------------------------- latent traits
    engagement = rng.beta(2.2, 2.2, n_subscribers)                 # baseline intensity 0..1
    trend = rng.normal(0.0, 0.32, n_subscribers)                   # + rising, - declining
    auto_renew = (rng.random(n_subscribers) < 0.72).astype(int)
    tenure_days = rng.gamma(shape=2.0, scale=330, size=n_subscribers).astype(int) + 30
    tenure_days = np.clip(tenure_days, 30, 2600)
    registration = expire_date - pd.to_timedelta(tenure_days, unit="D")

    plan_days = np.array(rng.choice(PLAN_DAYS, n_subscribers))
    plan_price = np.array([int(rng.choice(PLAN_PRICES[int(d)])) for d in plan_days])
    discount_pct = rng.choice([0.0, 0.0, 0.0, 0.05, 0.10], n_subscribers)
    n_prior_cancels = rng.poisson(0.18, n_subscribers)

    # ---------------------------------------------------------------- churn model
    # Logistic combination of behavioural drivers, standardised so coefficients
    # are interpretable, plus noise so the problem stays non-trivial.
    z = (
        -0.55
        - 2.10 * (engagement - 0.5)                       # engaged subscribers stay
        - 1.35 * trend                                    # declining engagement -> churn
        - 1.15 * auto_renew                               # auto-renew strongly reduces churn
        + 0.55 * n_prior_cancels                          # cancellation history
        + 0.45 * ((plan_price - 250) / 450)               # price pressure
        - 0.60 * ((tenure_days - 700) / 700)              # long tenure is sticky
        + rng.normal(0, 0.85, n_subscribers)              # irreducible noise
    )
    churn_prob = _sigmoid(z)
    is_churn = (rng.random(n_subscribers) < churn_prob).astype(int)

    members = pd.DataFrame({
        "msno": msnos,
        "city": rng.choice(CITIES, n_subscribers),
        "bd": np.where(rng.random(n_subscribers) < 0.35, 0,
                       np.clip(rng.normal(29, 9, n_subscribers).astype(int), 13, 72)),
        "gender": rng.choice(["male", "female", ""], n_subscribers, p=[0.42, 0.40, 0.18]),
        "registered_via": rng.choice(REGISTERED_VIA, n_subscribers, p=[.18, .12, .42, .22, .06]),
        "registration_init_time": registration.dt.strftime("%Y%m%d").astype(int),
    })

    # ---------------------------------------------------------------- transactions
    tx_rows = []
    for i in range(n_subscribers):
        exp_i = expire_date.iloc[i]
        pd_i = int(plan_days[i])
        price_i = int(plan_price[i])
        paid_i = int(round(price_i * (1 - discount_pct[i])))
        # subscription history walking backwards from the expiry date
        n_tx = int(np.clip(tenure_days[i] // max(pd_i, 1), 1, 14))
        cancels_left = int(n_prior_cancels[i])
        for k in range(n_tx):
            t_date = exp_i - pd.Timedelta(pd_i * (k + 1), unit="D")
            if t_date < registration.iloc[i]:
                break
            m_expire = t_date + pd.Timedelta(pd_i, unit="D")
            is_cancel = 0
            if cancels_left > 0 and k > 0 and rng.random() < 0.45:
                is_cancel = 1
                cancels_left -= 1
            tx_rows.append((
                msnos[i], int(rng.choice(PAYMENT_METHODS)), pd_i, price_i, paid_i,
                int(auto_renew[i]), int(t_date.strftime("%Y%m%d")),
                int(m_expire.strftime("%Y%m%d")), is_cancel,
            ))
        # the renewal that decides the label, written AFTER the cutoff
        if is_churn[i] == 0:
            renew = exp_i + pd.Timedelta(int(rng.integers(0, config.CHURN_WINDOW_DAYS - 1)), unit="D")
            tx_rows.append((
                msnos[i], int(rng.choice(PAYMENT_METHODS)), pd_i, price_i, paid_i,
                int(auto_renew[i]), int(renew.strftime("%Y%m%d")),
                int((renew + pd.Timedelta(pd_i, unit="D")).strftime("%Y%m%d")), 0,
            ))
        elif rng.random() < 0.28:
            # some churners cancel explicitly just before lapsing
            c_date = exp_i - pd.Timedelta(int(rng.integers(1, 10)), unit="D")
            tx_rows.append((
                msnos[i], int(rng.choice(PAYMENT_METHODS)), pd_i, price_i, paid_i,
                0, int(c_date.strftime("%Y%m%d")),
                int(exp_i.strftime("%Y%m%d")), 1,
            ))

    transactions = pd.DataFrame(tx_rows, columns=[
        "msno", "payment_method_id", "payment_plan_days", "plan_list_price",
        "actual_amount_paid", "is_auto_renew", "transaction_date",
        "membership_expire_date", "is_cancel",
    ])

    # ---------------------------------------------------------------- user logs
    # 120 days of listening history ending at each subscriber's own cutoff.
    HIST = 120
    log_frames = []
    for i in range(n_subscribers):
        exp_i = expire_date.iloc[i]
        base = engagement[i]
        # daily intensity ramps by the subscriber's trend across the history
        ramp = np.linspace(-1.0, 1.0, HIST) * trend[i]
        intensity = np.clip(base + ramp, 0.02, 1.0)
        active = rng.random(HIST) < (0.25 + 0.65 * intensity)
        k = int(active.sum())
        if k == 0:
            continue
        offsets = np.arange(HIST)[active]                   # 0 = oldest
        dates = exp_i - pd.to_timedelta(HIST - 1 - offsets, unit="D")
        inten = intensity[active]
        n100 = rng.poisson(np.maximum(inten * 26, 0.2))
        n25 = rng.poisson(np.maximum((1 - inten) * 11, 0.2))
        n50 = rng.poisson(np.maximum(inten * 4, 0.1))
        n75 = rng.poisson(np.maximum(inten * 3, 0.1))
        n985 = rng.poisson(np.maximum(inten * 3, 0.1))
        total = n25 + n50 + n75 + n985 + n100
        log_frames.append(pd.DataFrame({
            "msno": msnos[i],
            "date": dates.strftime("%Y%m%d").astype(int),
            "num_25": n25, "num_50": n50, "num_75": n75,
            "num_985": n985, "num_100": n100,
            "num_unq": np.maximum((total * rng.uniform(0.55, 0.9, k)).astype(int), 1),
            "total_secs": np.round(n100 * rng.normal(215, 28, k)
                                   + n25 * rng.normal(28, 8, k)
                                   + (n50 + n75 + n985) * rng.normal(120, 25, k), 2).clip(0),
        }))
    user_logs = pd.concat(log_frames, ignore_index=True)

    train = pd.DataFrame({"msno": msnos, "is_churn": is_churn})

    # cohort metadata: the observation cutoff per subscriber, used by the
    # feature builder and by the temporal split. Kept beside the raw files
    # because in the real dataset this is derived, not given.
    cohorts = pd.DataFrame({
        "msno": msnos,
        "cohort_month": cohort_month,
        "observation_cutoff": expire_date.dt.strftime("%Y%m%d").astype(int),
    })

    members.to_csv(config.RAW_FILES["members"], index=False)
    transactions.to_csv(config.RAW_FILES["transactions"], index=False)
    user_logs.to_csv(config.RAW_FILES["user_logs"], index=False)
    train.to_csv(config.RAW_FILES["train"], index=False)
    cohorts.to_csv(config.RAW_DIR / "cohorts.csv", index=False)

    return {"members": members, "transactions": transactions, "user_logs": user_logs,
            "train": train, "cohorts": cohorts}


def _summary(d: dict, seed: int) -> None:
    members, tx, logs, train, cohorts = (d["members"], d["transactions"], d["user_logs"],
                                         d["train"], d["cohorts"])
    tdate = pd.to_datetime(tx["transaction_date"].astype(str), format="%Y%m%d")
    ldate = pd.to_datetime(logs["date"].astype(str), format="%Y%m%d")

    print("=" * 56)
    print("Generated ChurnCast synthetic dataset")
    print("  (KKBOX-like structure - NOT the official KKBOX dataset)")
    print("=" * 56)
    print(f"Subscribers      : {len(members):,}")
    print(f"Transactions     : {len(tx):,}")
    print(f"User logs        : {len(logs):,}")
    print(f"Churn rate       : {train['is_churn'].mean():.4f}")
    print(f"Seed             : {seed}")
    print()
    print("Date ranges:")
    print(f"  transactions   : {tdate.min().date()} -> {tdate.max().date()}")
    print(f"  user_logs      : {ldate.min().date()} -> {ldate.max().date()}")
    print(f"  registrations  : {pd.to_datetime(members.registration_init_time.astype(str), format='%Y%m%d').min().date()}"
          f" -> {pd.to_datetime(members.registration_init_time.astype(str), format='%Y%m%d').max().date()}")
    print()
    print("Cohorts (observation cutoff month -> subscribers, churn rate):")
    m = cohorts.merge(train, on="msno")
    for month, g in m.groupby("cohort_month"):
        split = ("TRAIN" if month in config.TRAIN_COHORTS
                 else "VAL" if month in config.VAL_COHORTS else "TEST")
        print(f"  {month}  {len(g):>6,}   churn {g['is_churn'].mean():.4f}   [{split}]")
    print()
    print("Missing values per file:")
    for name, df in [("members", members), ("transactions", tx),
                     ("user_logs", logs), ("train", train)]:
        tot = int(df.isna().sum().sum())
        print(f"  {name:14s} {tot}")
    print()
    print("Sample subscriber IDs:")
    for s in members["msno"].head(5):
        print(f"  {s}")
    print("=" * 56)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-subscribers", type=int, default=config.N_SUBSCRIBERS)
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()
    data = generate(args.n_subscribers, args.seed)
    _summary(data, args.seed)


if __name__ == "__main__":
    main()
