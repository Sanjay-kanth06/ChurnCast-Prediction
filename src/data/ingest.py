"""
W1-T1: Kaggle ingestion.

Downloads the KKBOX Churn Prediction Challenge dataset into data/raw/ using
the Kaggle API. Requires KAGGLE_USERNAME / KAGGLE_KEY (env vars or
~/.kaggle/kaggle.json) with access to the competition.

Usage:
    python -m src.data.ingest
"""
from __future__ import annotations

import logging
import os
import zipfile
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ingest")

COMPETITION = "kkbox-churn-prediction-challenge"
RAW_DIR = Path("data/raw")
EXPECTED_FILES = ["train.csv", "transactions.csv", "user_logs.csv", "members.csv"]


def download(force: bool = False) -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if not force and all((RAW_DIR / f).exists() for f in EXPECTED_FILES):
        # The presence check is by filename only, so it cannot tell a correct
        # file from a wrong or fabricated one -- which is how a synthetic
        # user_logs.csv and the label-window transactions_v2 both survived here.
        # Re-run with --force to fetch regardless.
        log.info("All raw filenames already present in %s, skipping download.", RAW_DIR)
        log.info("Filenames match; contents are NOT verified. Use --force to re-download.")
        _log_row_counts()
        return

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as e:
        raise RuntimeError(
            "kaggle package not installed. Run `pip install kaggle` and configure "
            "credentials in ~/.kaggle/kaggle.json or KAGGLE_USERNAME / KAGGLE_KEY."
        ) from e

 

    api = KaggleApi()
    api.authenticate()

    log.info("Downloading competition files for %s ...", COMPETITION)
    api.competition_download_files(COMPETITION, path=str(RAW_DIR), quiet=False)

    zip_path = RAW_DIR / f"{COMPETITION}.zip"
    if zip_path.exists():
        log.info("Unzipping %s ...", zip_path)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(RAW_DIR)
        zip_path.unlink()

    # KKBOX ships some files gzipped (transactions.csv.7z / user_logs.csv.7z on
    # some mirrors) -- if present, note it rather than silently failing.
    missing = [f for f in EXPECTED_FILES if not (RAW_DIR / f).exists()]
    if missing:
        log.warning(
            "Missing expected files after extraction: %s. Check for nested "
            "archives (.7z) that need manual extraction.",
            missing,
        )

    _log_row_counts()


def _log_row_counts() -> None:
    import pandas as pd

    for f in EXPECTED_FILES:
        path = RAW_DIR / f
        if not path.exists():
            log.warning("%s not found, cannot log row count.", f)
            continue
        # count rows without loading the whole (potentially huge) file into memory
        with open(path, "rb") as fh:
            n = sum(1 for _ in fh) - 1  # minus header
        log.info("%-18s rows=%d", f, n)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="download even if the expected filenames already exist")
    args = parser.parse_args()
    download(force=args.force)
