"""
DEPRECATED - DO NOT RUN. Kept only as a record of how data/raw/user_logs.csv
came to be fabricated.

This script generates random listening history from `rng.random()` for a sample
of real member IDs and writes it to data/raw/user_logs.csv, OVERWRITING the real
KKBOX file. There is no undo: the original is not backed up anywhere.

Consequences of the last run, measured on the resulting model:
  - covers 15,592 of 970,960 labelled members (1.6%)
  - days_since_last_listen becomes a constant (single observed value)
  - all three engagement features score at chance (AUC 0.500, 0.5005, 0.5002)

To get the real file, re-download the competition data:
    python -m src.data.ingest --force
    python extract_kkbox.py
"""
import sys

import numpy as np
import pandas as pd
from pathlib import Path

sys.exit(
    "make_synthetic_user_logs.py is deprecated and will overwrite real data.\n"
    "See the module docstring. Use `python -m src.data.ingest --force` instead."
)

RAW_DIR = Path("data/raw")
rng = np.random.default_rng(42)

print("Reading real transactions.csv for msno list ...")
transactions = pd.read_csv(RAW_DIR / "transactions.csv", usecols=["msno"])
msnos = transactions["msno"].unique()
print(f"Found {len(msnos)} unique real members")

n_members = min(len(msnos), 20000)
sample_msnos = rng.choice(msnos, size=n_members, replace=False)
print(f"Generating synthetic logs for {n_members} of them ...")

log_rows = []
base = pd.Timestamp("2017-02-28")
for msno in sample_msnos:
    n_days = rng.integers(10, 120)
    engagement_level = rng.random()
    for d in range(n_days):
        date = base - pd.Timedelta(days=int(d))
        secs = max(0, rng.normal(3000 * engagement_level, 800))
        log_rows.append({
            "msno": msno,
            "date": int(date.strftime("%Y%m%d")),
            "num_25": int(rng.poisson(2 * engagement_level)),
            "num_50": int(rng.poisson(1 * engagement_level)),
            "num_75": int(rng.poisson(1 * engagement_level)),
            "num_985": int(rng.poisson(1 * engagement_level)),
            "num_100": int(rng.poisson(8 * engagement_level)),
            "num_unq": int(rng.poisson(6 * engagement_level)),
            "total_secs": round(secs, 2),
        })

user_logs = pd.DataFrame(log_rows)
user_logs.to_csv(RAW_DIR / "user_logs.csv", index=False)
print(f"Wrote user_logs.csv rows={len(user_logs)}")