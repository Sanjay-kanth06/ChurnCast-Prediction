"""
Extracts the KKBOX competition archives into data/raw/ under the names the
pipeline expects.

WHY THE MAPPING CHANGED
-----------------------
The previous version mapped `transactions_v2.csv.7z` -> `transactions.csv`.
That is the file that caused the leak: transactions_v2 is the *renewal window*
extract (1.43M rows, 75% of them March 2017), not the transaction history. Used
as feature input it hands the model the very renewals the label is computed
from, and at a correct feature cutoff of 2017-01-31 it leaves only 1.5% of
labelled members with any prior history at all.

The full-history file is plain `transactions.csv.7z` (~21.5M rows, 2015-01 to
2017-02). That is what features_v3 needs.

`train_v2.csv.7z` -> `train.csv` is correct and unchanged: train_v2 is the
February-2017 expiry cohort, which is what data/raw/WSDMChurnLabeller.scala
generates (historyCutoff 20170131, candidates expiring 20170201-20170228).
`train.csv.7z` (the January cohort) is extracted alongside as train_v1.csv for
anyone building the two-cohort temporal split.

Usage:
    python extract_kkbox.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import py7zr

RAW = Path("data/raw")

# archive name -> name the pipeline expects
MAPPING = {
    "transactions.csv.7z": "transactions.csv",   # v1 full history - NOT transactions_v2
    "user_logs.csv.7z": "user_logs.csv",         # v1, through 2017-02-28
    "members_v3.csv.7z": "members.csv",
    "train_v2.csv.7z": "train.csv",              # Feb-2017 expiry cohort
    "train.csv.7z": "train_v1.csv",              # Jan-2017 cohort, optional second split
}

# Extracted but deliberately not used as feature input - kept under their own
# names so nothing silently picks them up as history.
LABEL_WINDOW_ONLY = {
    "transactions_v2.csv.7z": "transactions_v2.csv",
    "user_logs_v2.csv.7z": "user_logs_v2.csv",
}


def extract(archive: str, target: str) -> bool:
    src = RAW / archive
    if not src.exists():
        print(f"  skip   {archive:26s} (not downloaded)")
        return False

    dest = RAW / target
    if dest.exists():
        backup = dest.with_suffix(dest.suffix + ".superseded")
        print(f"  moving existing {target} -> {backup.name}")
        dest.replace(backup)

    print(f"  extract {archive:26s} -> {target}")
    with py7zr.SevenZipFile(src, mode="r") as z:
        names = z.getnames()
        z.extractall(path=RAW)

    # the archive's inner filename rarely matches what we want it called
    for name in names:
        produced = RAW / name
        if produced.exists() and produced != dest:
            produced.replace(dest)
            break
    return dest.exists()


def main() -> None:
    if not RAW.exists():
        sys.exit(f"{RAW} does not exist - run `python -m src.data.ingest` first.")

    print("Feature-input files:")
    got = {t: extract(a, t) for a, t in MAPPING.items()}

    print("\nLabel-window files (extracted, not used as features):")
    for a, t in LABEL_WINDOW_ONLY.items():
        extract(a, t)

    required = ["transactions.csv", "user_logs.csv", "members.csv", "train.csv"]
    missing = [f for f in required if not (RAW / f).exists()]
    print()
    if missing:
        sys.exit(f"MISSING after extraction: {missing}\n"
                 "Check that the competition download completed and the .7z files are in data/raw/.")
    for f in required:
        size_gb = (RAW / f).stat().st_size / 1e9
        print(f"  {f:22s} {size_gb:7.2f} GB")
    print("\nNext: python -m src.data.features_v3")


if __name__ == "__main__":
    main()
