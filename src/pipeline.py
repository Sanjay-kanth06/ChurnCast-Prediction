"""
Reusable pipeline steps.

Both `scripts/run_pipeline.py` and the Airflow DAG call these functions, so the
business logic lives in exactly one place and the DAG stays a thin orchestration
layer.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from src import config

log = logging.getLogger("pipeline")


def step_ingest(n_subscribers: int | None = None, seed: int | None = None,
                force: bool = False) -> dict[str, Any]:
    """Ensures a raw dataset exists. Generates the synthetic one if absent."""
    from scripts.generate_sample_data import generate

    present = all(p.exists() for p in config.RAW_FILES.values())
    if present and not force:
        log.info("raw data already present in %s - skipping generation", config.RAW_DIR)
        import pandas as pd
        train = pd.read_csv(config.RAW_FILES["train"])
        return {"generated": False, "subscribers": int(len(train)),
                "churn_rate": float(train["is_churn"].mean())}

    data = generate(n_subscribers or config.N_SUBSCRIBERS, seed or config.SEED)
    return {"generated": True,
            "subscribers": int(len(data["train"])),
            "transactions": int(len(data["transactions"])),
            "user_logs": int(len(data["user_logs"])),
            "churn_rate": float(data["train"]["is_churn"].mean())}


def step_build_features() -> dict[str, Any]:
    from src.features import build_feature_table
    table = build_feature_table()
    return {"rows": int(len(table)),
            "features": int(len(table.columns) - 2),
            "churn_rate": float(table["is_churn"].mean()),
            "path": str(config.FEATURE_TABLE_CSV)}


def step_train() -> dict[str, Any]:
    """Trains both candidates, evaluates on the held-out cohort, writes
    artifacts and applies the promotion gate."""
    from src.models.train import main as train_main
    return train_main()


def step_evaluate() -> dict[str, Any]:
    """Reads back the metrics the training step persisted. Separated from
    training so the DAG can gate on it independently."""
    path = config.REPORTS_DIR / "model_metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run the train step first")
    payload = json.loads(path.read_text(encoding="utf-8"))
    best = payload["best_model"]
    log.info("evaluate | best=%s val_roc_auc=%.4f test_roc_auc=%.4f", best,
             payload["validation"][best]["roc_auc"], payload["test"][best]["roc_auc"])
    return payload


def step_register_if_better() -> dict[str, Any]:
    """Reports the registry decision made during training.

    Registration happens inside the training step because that is where the
    fitted model object lives. This step surfaces the outcome so the DAG has an
    explicit, inspectable gate task.
    """
    payload = step_evaluate()
    decision = payload.get("registry_decision", "")
    version = payload.get("registered_version")
    if version is None:
        log.warning("no new version registered: %s", decision)
    else:
        log.info("registered version %s", version)
    return {"registered_version": version, "decision": decision}


def step_drift_report(mode: str = "demo") -> dict[str, Any]:
    from src.features import load_feature_table
    from src.monitoring.drift import (
        generate_drift_report, inject_synthetic_drift, split_reference_current,
    )
    table = load_feature_table()
    reference, current = split_reference_current(table)
    if mode == "demo":
        current = inject_synthetic_drift(current)
    return generate_drift_report(reference, current)
