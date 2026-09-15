"""
Runs the complete ChurnCast prototype pipeline end to end.

    python scripts/run_pipeline.py [--force-data] [--skip-drift]

Steps: ingest -> build_features -> train -> evaluate -> register_if_better,
optionally followed by a drift report.
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import config, pipeline  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("run_pipeline")

LINE = "=" * 60


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force-data", action="store_true",
                    help="regenerate the synthetic dataset even if it exists")
    ap.add_argument("--skip-drift", action="store_true")
    args = ap.parse_args()

    config.ensure_dirs()
    t0 = time.time()

    ingest = pipeline.step_ingest(force=args.force_data)
    feats = pipeline.step_build_features()
    metrics = pipeline.step_train()
    pipeline.step_evaluate()
    registry = pipeline.step_register_if_better()

    drift = None
    if not args.skip_drift:
        try:
            drift = pipeline.step_drift_report(mode="demo")
        except Exception as e:                                # pragma: no cover
            log.warning("drift report failed: %s", e)

    elapsed = time.time() - t0
    best = metrics["best_model"]
    val = metrics["validation"]
    test = metrics["test"][best]

    out = []
    out.append(LINE)
    out.append("ChurnCast Prototype Pipeline")
    out.append(LINE)
    out.append("")
    out.append("Dataset:")
    out.append("  KKBOX-like synthetic dataset (NOT official KKBOX data)")
    out.append(f"  Subscribers        : {ingest['subscribers']:,}")
    out.append(f"  Churn rate         : {ingest['churn_rate']:.4f}")
    out.append(f"  Seed               : {config.SEED}")
    out.append("")
    out.append("Features:")
    out.append(f"  Feature count      : {feats['features']}")
    out.append(f"  Feature table      : {feats['path']}")
    out.append("")
    out.append("Temporal split (by observation-cutoff cohort):")
    s = metrics["split"]
    out.append(f"  Train  {s['train_cohorts']}  n={s['n_train']:,}")
    out.append(f"  Val    {s['val_cohorts']}  n={s['n_val']:,}")
    out.append(f"  Test   {s['test_cohorts']}  n={s['n_test']:,}")
    out.append("")
    out.append("Validation metrics (used for model selection):")
    for name, m in val.items():
        out.append(f"  {name:22s} ROC-AUC {m['roc_auc']:.4f}  PR-AUC {m['pr_auc']:.4f}  "
                   f"F1 {m['f1']:.4f}")
    out.append("")
    out.append(f"Best model: {best}")
    out.append("")
    out.append("Test metrics (held-out cohort, scored once):")
    out.append(f"  ROC-AUC {test['roc_auc']:.4f}  PR-AUC {test['pr_auc']:.4f}  "
               f"Precision {test['precision']:.4f}  Recall {test['recall']:.4f}  "
               f"F1 {test['f1']:.4f}")
    out.append("")
    out.append("MLflow:")
    out.append(f"  Experiment         : {config.MLFLOW_EXPERIMENT}")
    out.append(f"  Run ID             : {metrics['best_run_id']}")
    out.append(f"  Registered model   : {config.MLFLOW_MODEL_NAME}")
    out.append(f"  Version            : {registry['registered_version']}")
    out.append(f"  Registry decision  : {registry['decision']}")
    if drift:
        out.append("")
        out.append("Drift report (synthetic drift demonstration):")
        out.append(f"  Drifted features   : {drift['drifted_features']}/{drift['n_features']}")
        out.append(f"  Report             : {drift['output_path']}")
    out.append("")
    out.append("Artifacts written to reports/:")
    for f in sorted(config.REPORTS_DIR.glob("*")):
        if f.is_file() and f.suffix in {".png", ".csv", ".json", ".txt", ".html"}:
            out.append(f"  {f.name}")
    out.append("")
    out.append(f"Elapsed: {elapsed:.1f}s")
    out.append(LINE)
    out.append("Pipeline completed successfully")
    out.append(LINE)

    text = "\n".join(out)
    print(text)
    (config.REPORTS_DIR / "pipeline_summary.txt").write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
