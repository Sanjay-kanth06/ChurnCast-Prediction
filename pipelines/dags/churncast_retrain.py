"""
ChurnCast weekly retraining DAG.

    ingest -> build_features -> train -> evaluate -> register_if_better

Every task is a thin wrapper over a function in `src.pipeline`, so the DAG
carries orchestration concerns only (scheduling, retries, dependencies) and no
business logic. The same functions back `scripts/run_pipeline.py`, which means
the pipeline can be exercised without an Airflow installation.

The promotion gate lives in `register_if_better`: a newly trained model replaces
the registered one only when it improves validation ROC-AUC. Otherwise the
existing model stays in place and the task logs the decision.

Airflow is not installed in the host Python environment (no release supports
Python 3.13 at the time of writing); this DAG is intended to run in the Airflow
container defined in docker-compose.yml.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

# Make `src` importable when Airflow loads this file from the dags folder.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from airflow import DAG  # noqa: E402
from airflow.operators.python import PythonOperator  # noqa: E402

from src import pipeline  # noqa: E402

DEFAULT_ARGS = {
    "owner": "churncast",
    "retries": 2,
    "retry_delay": timedelta(minutes=2),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=15),
    "depends_on_past": False,
}


def _ingest(**_):
    return pipeline.step_ingest()


def _build_features(**_):
    return pipeline.step_build_features()


def _train(**_):
    result = pipeline.step_train()
    # keep the XCom payload small: the full object holds fitted estimators
    return {"best_model": result["best_model"], "best_run_id": result["best_run_id"]}


def _evaluate(**_):
    payload = pipeline.step_evaluate()
    best = payload["best_model"]
    return {"best_model": best,
            "val_roc_auc": payload["validation"][best]["roc_auc"],
            "test_roc_auc": payload["test"][best]["roc_auc"]}


def _register_if_better(**_):
    return pipeline.step_register_if_better()


with DAG(
    dag_id="churncast_retrain",
    description="Weekly ChurnCast retraining with a validation-ROC-AUC promotion gate",
    default_args=DEFAULT_ARGS,
    schedule="@weekly",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    max_active_runs=1,
    tags=["churncast", "mlops", "prototype"],
) as dag:

    ingest = PythonOperator(task_id="ingest", python_callable=_ingest)
    build_features = PythonOperator(task_id="build_features", python_callable=_build_features)
    train = PythonOperator(task_id="train", python_callable=_train)
    evaluate = PythonOperator(task_id="evaluate", python_callable=_evaluate)
    register_if_better = PythonOperator(task_id="register_if_better",
                                        python_callable=_register_if_better)

    ingest >> build_features >> train >> evaluate >> register_if_better
