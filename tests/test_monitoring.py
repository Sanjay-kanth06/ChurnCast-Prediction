"""Evidently drift monitoring and MLflow tracking integration."""
from __future__ import annotations

import json

import pytest

from src import config


# --------------------------------------------------------------- drift

def test_reference_and_current_split(feature_table):
    from src.monitoring.drift import split_reference_current
    ref, cur = split_reference_current(feature_table)
    assert len(ref) > 0 and len(cur) > 0
    assert set(ref.index).isdisjoint(cur.index)


def test_inject_synthetic_drift_changes_distribution(feature_table):
    from src.monitoring.drift import inject_synthetic_drift, split_reference_current
    _, cur = split_reference_current(feature_table)
    shifted = inject_synthetic_drift(cur)
    assert shifted["recency_days"].mean() > cur["recency_days"].mean()
    assert shifted["active_days_last_30d"].mean() < cur["active_days_last_30d"].mean()
    assert len(shifted) == len(cur)


def test_drift_report_generates_html(feature_table, tmp_path):
    """The report must actually be produced by code, not asserted."""
    from src.monitoring.drift import generate_drift_report, split_reference_current
    ref, cur = split_reference_current(feature_table)
    out = tmp_path / "drift.html"
    summary = generate_drift_report(ref, cur, output_path=out)

    assert out.exists() and out.stat().st_size > 1000
    assert "<html" in out.read_text(encoding="utf-8", errors="ignore").lower()
    assert summary["n_features"] > 0


def test_drift_is_detected_when_injected(feature_table, tmp_path):
    from src.monitoring.drift import (
        generate_drift_report, inject_synthetic_drift, split_reference_current,
    )
    ref, cur = split_reference_current(feature_table)
    summary = generate_drift_report(ref, inject_synthetic_drift(cur),
                                    output_path=tmp_path / "drift_demo.html")
    assert summary["drifted_features"] > 0, "injected drift went undetected"


def test_persisted_drift_summary_if_present():
    path = config.REPORTS_DIR / "drift_summary_demo.json"
    if not path.exists():
        pytest.skip("drift report not generated yet")
    d = json.loads(path.read_text(encoding="utf-8"))
    assert d["mode"] == "demo"
    assert d["drifted_features"] >= 0
    assert "note" in d, "demo mode must be labelled as synthetic"


# --------------------------------------------------------------- mlflow

def test_mlflow_experiment_exists():
    import mlflow
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    exp = mlflow.get_experiment_by_name(config.MLFLOW_EXPERIMENT)
    if exp is None:
        pytest.skip("pipeline not run yet")
    assert exp.name == config.MLFLOW_EXPERIMENT


def test_training_created_runs_with_metrics():
    import mlflow
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    exp = mlflow.get_experiment_by_name(config.MLFLOW_EXPERIMENT)
    if exp is None:
        pytest.skip("pipeline not run yet")
    runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
    assert len(runs) >= 2, "expected a run per candidate model"
    assert "metrics.val_roc_auc" in runs.columns


def test_model_is_registered():
    import mlflow
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions(f"name='{config.MLFLOW_MODEL_NAME}'")
    if not versions:
        pytest.skip("pipeline not run yet")
    assert max(int(v.version) for v in versions) >= 1


def test_register_if_better_rejects_a_worse_candidate():
    """The promotion gate must not register a model that loses to the incumbent."""
    import mlflow
    from src.models.train import register_if_better, registered_best_roc_auc

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    incumbent = registered_best_roc_auc(client, config.MLFLOW_MODEL_NAME)
    if incumbent is None:
        pytest.skip("no registered model yet")

    worse = {"name": "deliberately_worse", "run_id": "does-not-matter",
             "val_metrics": {"roc_auc": incumbent - 0.05}}
    version, decision = register_if_better(worse)
    assert version is None
    assert "did not beat" in decision
