"""
Stage 4: fit the selected real model, log it to MLflow and register it.

Registered under its own name, ChurnCastRealV2, in its own experiment. The
synthetic ChurnCastModel is never opened, re-registered or promoted -- the two
live side by side in the same tracking store and the API serves whichever the
caller asks for.

The model is the Stage 3 winner: XGBoost, with_expiry, test ROC-AUC 0.9181. It
is refitted here rather than pickled out of Stage 3, because Stage 3 was an
evaluation harness and kept its models in memory. The split is reproduced from
the same seed and the same helper, so the run logged here is the run that was
evaluated.

TRAIN/SERVE ALIGNMENT
---------------------
The feature column order is written into the run as an artifact and re-read at
serving time. This is not ceremony: the synthetic pipeline shipped a train/serve
skew twice, once from categorical dtype round-tripping and once from a bare
read_csv that turned code columns into ints during training and strings at
serving. A column list that travels with the model is what makes that class of
bug loud instead of silent.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime

import mlflow
from mlflow import MlflowClient

from src import config
from src.models.train_real_v2 import (
    CATEGORICAL, DATA_DIR, EXPIRY_FEATURE, FEATURES_PARQUET, PLOTS_DIR, SEED,
    build_models, evaluate, importances, load, split,
)

log = logging.getLogger("register_real_v2")

EXPERIMENT = "ChurnCastRealV2"
MODEL_NAME = "ChurnCastRealV2"
SELECTED_MODEL = "xgboost"
SELECTED_VARIANT = "with_expiry"

OBSERVATION_CUTOFF = "2017-02-28"
DATA_MODE = "real_v2"

FEATURE_COLUMNS_JSON = DATA_DIR / "model_feature_columns.json"
REGISTRATION_JSON = DATA_DIR / "registration_real_v2.json"


def fit_and_register(write: bool = True) -> dict:
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    X, y = load()
    columns = list(X.columns)          # order matters; persisted below
    parts = split(X, y)
    X_train, y_train = parts["train"]
    X_val, y_val = parts["val"]
    X_test, y_test = parts["test"]

    model = build_models(columns)[SELECTED_MODEL]

    t0 = time.time()
    with mlflow.start_run(run_name=f"{SELECTED_MODEL}_{SELECTED_VARIANT}") as run:
        model.fit(X_train, y_train)
        fit_seconds = round(time.time() - t0, 1)

        val_metrics = evaluate(model, X_val, y_val)
        test_metrics = evaluate(model, X_test, y_test)
        top = importances(SELECTED_MODEL, model, columns, top=30)

        clf = model.named_steps["clf"]
        mlflow.log_params({
            "model_family": SELECTED_MODEL,
            "variant": SELECTED_VARIANT,
            "seed": SEED,
            "n_estimators": clf.n_estimators,
            "max_depth": clf.max_depth,
            "learning_rate": clf.learning_rate,
            "subsample": clf.subsample,
            "colsample_bytree": clf.colsample_bytree,
            "scale_pos_weight": round(float(clf.scale_pos_weight), 4),
            "tree_method": "hist",
            "feature_count": len(columns),
            "categorical_features": ",".join(CATEGORICAL),
            "expiry_feature_included": EXPIRY_FEATURE in columns,
            "observation_cutoff": OBSERVATION_CUTOFF,
            "data_mode": DATA_MODE,
            "split_strategy": "stratified random 60/20/20",
            "label_window": "March 2017 (train_v2)",
        })

        mlflow.log_metrics({
            "n_train": len(y_train), "n_val": len(y_val), "n_test": len(y_test),
            "train_positives": int(y_train.sum()),
            "val_positives": int(y_val.sum()),
            "test_positives": int(y_test.sum()),
            "positive_rate": float(y.mean()),
            "fit_seconds": fit_seconds,
        })
        for split_name, m in (("val", val_metrics), ("test", test_metrics)):
            mlflow.log_metrics({
                f"{split_name}_{k}": v for k, v in m.items()
                if isinstance(v, (int, float))})
            cm = m["confusion_matrix"]
            mlflow.log_metrics({f"{split_name}_cm_{k}": v for k, v in cm.items()})

        # Artifacts: the column contract, the importances, the plots.
        FEATURE_COLUMNS_JSON.parent.mkdir(parents=True, exist_ok=True)
        contract = {
            "model_name": MODEL_NAME,
            "feature_columns": columns,
            "categorical_features": CATEGORICAL,
            "dtypes": {c: str(X[c].dtype) for c in columns},
            "observation_cutoff": OBSERVATION_CUTOFF,
            "data_mode": DATA_MODE,
        }
        FEATURE_COLUMNS_JSON.write_text(json.dumps(contract, indent=2),
                                        encoding="utf-8")
        mlflow.log_artifact(str(FEATURE_COLUMNS_JSON))

        imp_path = DATA_DIR / "_feature_importance.json"
        imp_path.write_text(json.dumps(top, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(imp_path))
        imp_path.unlink(missing_ok=True)

        if PLOTS_DIR.exists():
            for png in sorted(PLOTS_DIR.glob("*.png")):
                mlflow.log_artifact(str(png), artifact_path="plots")

        mlflow.set_tags({
            "data_source": "KKBOX WSDM Cup 2018 (real)",
            "train_v2": str(DATA_DIR.parent.parent / "raw_kkbox" / "train_v2"),
            "features_parquet": str(FEATURES_PARQUET),
            "population": "970960",
            "scope_note": ("Supervised on official train_v2 labels with a "
                           "documented 2017-02-28 cutoff. Not a recreation of "
                           "the competition cohort-selection artifact."),
            "expiry_sensitivity": "passed (roc delta 0.002676, importance 0.85%)",
            "synthetic_model_untouched": "ChurnCastModel",
        })

        # Same serialization the synthetic pipeline uses: cloudpickle, because
        # the default skops format rejects the estimator internals.
        mlflow.sklearn.log_model(model, name="model",
                                 input_example=X_train.head(3),
                                 serialization_format="cloudpickle")
        run_id = run.info.run_id

    mv = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)

    client = MlflowClient()
    version = client.get_model_version(MODEL_NAME, mv.version)
    # Wait for the version to leave PENDING_REGISTRATION.
    for _ in range(30):
        if version.status == "READY":
            break
        time.sleep(1)
        version = client.get_model_version(MODEL_NAME, mv.version)

    result = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "experiment": EXPERIMENT,
        "run_id": run_id,
        "model_name": MODEL_NAME,
        "model_version": mv.version,
        "status": version.status,
        "tracking_uri": config.MLFLOW_TRACKING_URI,
        "selected": {"family": SELECTED_MODEL, "variant": SELECTED_VARIANT},
        "feature_count": len(columns),
        "fit_seconds": fit_seconds,
        "validation": val_metrics,
        "test": test_metrics,
        "top_features": top[:15],
        "synthetic_model_name": config.MLFLOW_MODEL_NAME,
    }
    if write:
        REGISTRATION_JSON.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log.info("registered %s v%s (%s) run=%s", MODEL_NAME, mv.version,
             version.status, run_id)
    return result


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = fit_and_register()
    print(json.dumps({k: result[k] for k in (
        "experiment", "run_id", "model_name", "model_version", "status",
        "feature_count", "fit_seconds")}, indent=2))
    print("test:", json.dumps({k: round(v, 4) for k, v in result["test"].items()
                               if isinstance(v, float)}, indent=2))
    return 0 if result["status"] == "READY" else 1


if __name__ == "__main__":
    raise SystemExit(main())
