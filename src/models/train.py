"""
ChurnCast prototype training: temporal split, two models, MLflow tracking and
registry promotion.

EVALUATION PROTOCOL
-------------------
The split is temporal, not random. Each subscriber belongs to a monthly cohort
defined by their own observation cutoff, and the cohorts are assigned whole:

    train      2024-07 .. 2024-10      earlier cutoffs
    validation 2024-11                 later
    test       2024-12                 latest

Model selection uses VALIDATION ROC-AUC only. The winning model is scored ONCE
on the test cohort, and that number is never used to choose anything. This keeps
the test cohort an honest estimate of forward-in-time generalisation.

Usage:
    python -m src.models.train
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mlflow  # noqa: E402
import mlflow.sklearn  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.pipeline import Pipeline  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from src import config  # noqa: E402
from src.features import (  # noqa: E402
    FEATURE_COLUMNS, build_feature_table, build_preprocessor,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("train")


# ------------------------------------------------------------------ split

def time_based_split(table: pd.DataFrame):
    """Split by cohort month so that evaluation always looks forward in time.

    Returns X_train, X_val, X_test, y_train, y_val, y_test.
    """
    if "cohort_month" not in table.columns:
        raise ValueError("feature table has no cohort_month; cannot split temporally")

    def part(months):
        sub = table[table["cohort_month"].isin(months)]
        return sub[FEATURE_COLUMNS], sub["is_churn"]

    X_train, y_train = part(config.TRAIN_COHORTS)
    X_val, y_val = part(config.VAL_COHORTS)
    X_test, y_test = part(config.TEST_COHORTS)

    for name, X in [("train", X_train), ("val", X_val), ("test", X_test)]:
        if X.empty:
            raise ValueError(f"{name} partition is empty - check cohort configuration")

    log.info("temporal split | train %s=%d  val %s=%d  test %s=%d",
             config.TRAIN_COHORTS, len(X_train),
             config.VAL_COHORTS, len(X_val),
             config.TEST_COHORTS, len(X_test))
    return X_train, X_val, X_test, y_train, y_val, y_test


# ------------------------------------------------------------------ metrics

def evaluate(y_true, y_prob, threshold: float = config.DECISION_THRESHOLD) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def _plot_confusion(y_true, y_pred, path: Path, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.2, 3.8))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black", fontsize=11)
    ax.set_xticks([0, 1], ["Stay", "Churn"])
    ax.set_yticks([0, 1], ["Stay", "Churn"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_roc(curves: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    for name, (y_true, y_prob) in curves.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        ax.plot(fpr, tpr, lw=1.8, label=f"{name} (AUC={roc_auc_score(y_true, y_prob):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.9, label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _plot_pr(curves: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    for name, (y_true, y_prob) in curves.items():
        pr, rc, _ = precision_recall_curve(y_true, y_prob)
        ax.plot(rc, pr, lw=1.8,
                label=f"{name} (AP={average_precision_score(y_true, y_prob):.3f})")
    base = float(np.mean(list(curves.values())[0][0]))
    ax.axhline(base, ls="--", c="k", lw=0.9, label=f"base rate={base:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ------------------------------------------------------------------ models

def build_models(y_train: pd.Series) -> dict:
    """Both candidates handle imbalance through class weighting rather than
    resampling: synthetic minority oversampling would create examples that never
    existed in any cohort, which undermines the temporal guarantee of the split."""
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    scale_pos_weight = neg / max(pos, 1)

    return {
        "logistic_regression": (
            LogisticRegression(max_iter=2000, class_weight="balanced",
                               random_state=config.SEED),
            {"max_iter": 2000, "class_weight": "balanced"},
        ),
        "gradient_boosting": (
            XGBClassifier(
                n_estimators=400, max_depth=4, learning_rate=0.05,
                subsample=0.85, colsample_bytree=0.85, reg_lambda=1.2,
                eval_metric="auc", scale_pos_weight=scale_pos_weight,
                random_state=config.SEED, n_jobs=4,
            ),
            {"n_estimators": 400, "max_depth": 4, "learning_rate": 0.05,
             "subsample": 0.85, "colsample_bytree": 0.85, "reg_lambda": 1.2,
             "scale_pos_weight": round(scale_pos_weight, 4)},
        ),
    }


def train_one(name, estimator, params, X_train, y_train, X_val, y_val, n_test: int):
    """Fits, calibrates, evaluates on validation, and logs one MLflow run."""
    with mlflow.start_run(run_name=name) as run:
        mlflow.set_tags({"model_type": name, "dataset": "synthetic-kkbox-like",
                         "split": "temporal"})
        mlflow.log_params(params)
        mlflow.log_params({
            "seed": config.SEED, "feature_count": len(FEATURE_COLUMNS),
            "n_train": len(X_train), "n_val": len(X_val), "n_test": n_test,
            "calibrated": True, "calibration_method": "sigmoid",
            "decision_threshold": config.DECISION_THRESHOLD,
        })

        pipe = Pipeline([("preprocess", build_preprocessor()), ("clf", estimator)])
        model = CalibratedClassifierCV(pipe, method="sigmoid", cv=3)
        model.fit(X_train, y_train)

        val_prob = model.predict_proba(X_val)[:, 1]
        val_metrics = evaluate(y_val, val_prob)
        mlflow.log_metrics({f"val_{k}": v for k, v in val_metrics.items()})

        # cloudpickle rather than the skops default: skops refuses to serialise
        # CalibratedClassifierCV's internals as "untrusted types"
        mlflow.sklearn.log_model(model, name="model", input_example=X_train.head(3),
                                 serialization_format="cloudpickle")

        log.info("%-20s VAL roc_auc=%.4f pr_auc=%.4f f1=%.4f",
                 name, val_metrics["roc_auc"], val_metrics["pr_auc"], val_metrics["f1"])
        return {"name": name, "run_id": run.info.run_id, "model": model,
                "val_metrics": val_metrics, "val_prob": val_prob}


# ------------------------------------------------------------------ registry

def registered_best_roc_auc(client, model_name: str) -> float | None:
    """Validation ROC-AUC of the currently registered model, or None if there
    is no registered version yet."""
    try:
        versions = client.search_model_versions(f"name='{model_name}'")
    except Exception:
        return None
    best = None
    for v in versions:
        try:
            r = client.get_run(v.run_id)
        except Exception:
            continue
        m = r.data.metrics.get("val_roc_auc")
        if m is not None and (best is None or m > best):
            best = m
    return best


def register_if_better(result: dict, model_name: str = config.MLFLOW_MODEL_NAME):
    """Registers the candidate only if it beats the incumbent on validation
    ROC-AUC. Returns (version_or_None, decision_string)."""
    client = mlflow.tracking.MlflowClient()
    incumbent = registered_best_roc_auc(client, model_name)
    candidate = result["val_metrics"]["roc_auc"]

    if incumbent is not None and candidate <= incumbent:
        msg = (f"candidate val_roc_auc={candidate:.4f} did not beat incumbent "
               f"{incumbent:.4f}; keeping existing model")
        log.warning(msg)
        return None, msg

    mv = mlflow.register_model(f"runs:/{result['run_id']}/model", model_name)
    msg = (f"registered {model_name} v{mv.version} "
           f"(val_roc_auc={candidate:.4f}"
           + (f" > incumbent {incumbent:.4f})" if incumbent is not None else ", first version)"))
    log.info(msg)
    return mv.version, msg


# ------------------------------------------------------------------ main

def main(rebuild_features: bool = False) -> dict:
    config.ensure_dirs()
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)

    if rebuild_features or not config.FEATURE_TABLE_CSV.exists():
        table = build_feature_table()
    else:
        table = pd.read_csv(config.FEATURE_TABLE_CSV, index_col="msno")

    X_train, X_val, X_test, y_train, y_val, y_test = time_based_split(table)

    results = []
    for name, (est, params) in build_models(y_train).items():
        results.append(train_one(name, est, params, X_train, y_train,
                                 X_val, y_val, len(X_test)))

    best = max(results, key=lambda r: r["val_metrics"]["roc_auc"])
    log.info("best on validation: %s (roc_auc=%.4f)", best["name"],
             best["val_metrics"]["roc_auc"])

    # ---- single scoring pass on the held-out test cohort --------------------
    test_prob = best["model"].predict_proba(X_test)[:, 1]
    test_metrics = evaluate(y_test, test_prob)
    test_pred = (test_prob >= config.DECISION_THRESHOLD).astype(int)
    log.info("%-20s TEST roc_auc=%.4f pr_auc=%.4f f1=%.4f", best["name"],
             test_metrics["roc_auc"], test_metrics["pr_auc"], test_metrics["f1"])

    # ---- artifacts ---------------------------------------------------------
    R = config.REPORTS_DIR
    _plot_confusion(y_test, test_pred, R / "confusion_matrix.png",
                    f"{best['name']} - test cohort {config.TEST_COHORTS[0]}")
    _plot_roc({r["name"]: (y_val, r["val_prob"]) for r in results} |
              {f"{best['name']} (test)": (y_test, test_prob)}, R / "roc_curve.png")
    _plot_pr({r["name"]: (y_val, r["val_prob"]) for r in results} |
             {f"{best['name']} (test)": (y_test, test_prob)}, R / "precision_recall_curve.png")

    report_txt = classification_report(y_test, test_pred, target_names=["Stay", "Churn"])
    (R / "classification_report.txt").write_text(report_txt, encoding="utf-8")

    comparison = pd.DataFrame([
        {"model": r["name"], "split": "validation", **r["val_metrics"]} for r in results
    ] + [{"model": best["name"], "split": "test", **test_metrics}])
    comparison.to_csv(R / "model_comparison.csv", index=False)

    # ---- registry ----------------------------------------------------------
    version, decision = register_if_better(best)

    metrics_payload = {
        "dataset": "synthetic KKBOX-like (prototype)",
        "seed": config.SEED,
        "n_subscribers": int(len(table)),
        "feature_count": len(FEATURE_COLUMNS),
        "split": {"train_cohorts": config.TRAIN_COHORTS, "val_cohorts": config.VAL_COHORTS,
                  "test_cohorts": config.TEST_COHORTS,
                  "n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test)},
        "validation": {r["name"]: r["val_metrics"] for r in results},
        "test": {best["name"]: test_metrics},
        "best_model": best["name"],
        "best_run_id": best["run_id"],
        "registered_model": config.MLFLOW_MODEL_NAME,
        "registered_version": version,
        "registry_decision": decision,
    }
    (R / "model_metrics.json").write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    # attach artifacts to the winning run
    with mlflow.start_run(run_id=best["run_id"]):
        mlflow.log_metrics({f"test_{k}": v for k, v in test_metrics.items()})
        for f in ["confusion_matrix.png", "roc_curve.png", "precision_recall_curve.png",
                  "classification_report.txt", "model_comparison.csv", "model_metrics.json"]:
            mlflow.log_artifact(str(R / f))

    return metrics_payload


if __name__ == "__main__":
    main()
