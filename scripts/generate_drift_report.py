"""
Generates an Evidently data-drift report for the ChurnCast prototype.

Two modes:

  --mode natural  (default)
      reference = training cohorts, current = latest cohort, as produced by the
      generator. Any drift reported here is whatever the synthetic process
      actually produced between cohorts.

  --mode demo
      the current cohort is additionally shifted by inject_synthetic_drift().
      This is a SYNTHETIC DRIFT DEMONSTRATION used to show that the monitoring
      pipeline detects change. It is not observed production drift.

Usage:
    python scripts/generate_drift_report.py --mode demo
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config  # noqa: E402
from src.features import load_feature_table  # noqa: E402
from src.monitoring.drift import (  # noqa: E402
    generate_drift_report, inject_synthetic_drift, split_reference_current,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("drift-report")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["natural", "demo"], default="natural")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    table = load_feature_table()
    reference, current = split_reference_current(table)
    log.info("reference=%d rows (%s)  current=%d rows (%s)",
             len(reference), config.TRAIN_COHORTS, len(current), config.TEST_COHORTS)

    label = ""
    if args.mode == "demo":
        current = inject_synthetic_drift(current)
        label = "_synthetic_drift_demo"
        log.warning("SYNTHETIC DRIFT DEMONSTRATION: the current dataset has been "
                    "deliberately shifted. This is not observed production drift.")

    out = args.out or (config.REPORTS_DIR /
                       f"drift_report_{date.today().isoformat()}{label}.html")
    summary = generate_drift_report(reference, current, output_path=out)

    summary["mode"] = args.mode
    summary["reference_rows"] = int(len(reference))
    summary["current_rows"] = int(len(current))
    summary["reference_cohorts"] = config.TRAIN_COHORTS
    summary["current_cohorts"] = config.TEST_COHORTS
    if args.mode == "demo":
        summary["note"] = ("Synthetic drift demonstration - the current dataset was "
                           "deliberately shifted. Not observed production drift.")

    (config.REPORTS_DIR / f"drift_summary_{args.mode}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")

    print("=" * 56)
    print(f"Drift report ({args.mode} mode)")
    print("=" * 56)
    print(f"Reference rows : {summary['reference_rows']:,}  {config.TRAIN_COHORTS}")
    print(f"Current rows   : {summary['current_rows']:,}  {config.TEST_COHORTS}")
    print(f"Features       : {summary['n_features']}")
    print(f"Drifted        : {summary['drifted_features']}")
    print(f"HTML report    : {summary['output_path']}")
    print("=" * 56)


if __name__ == "__main__":
    main()
