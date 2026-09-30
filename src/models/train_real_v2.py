"""
Real Stage 3: train and evaluate on the train_v2 population.

THE SPLIT, AND WHY IT IS NOT TEMPORAL
-------------------------------------
The synthetic pipeline splits by cohort month, because there subscribers are
observed at different times. That is impossible here and pretending otherwise
would be theatre: every one of the 970,960 subscribers shares a single
observation cutoff (2017-02-28) and a single label window (March 2017). There
is no time axis left inside the dataset to hold out along.

So the split is stratified random, 60/20/20, seed 42. What that measures is
generalisation to unseen subscribers in the same period -- not to a future
period. That is a real limitation and is recorded in the results rather than
glossed over. Leakage is controlled upstream by the cutoff enforced in Stage 2,
not by the split. Each msno appears exactly once, so there is no group leakage
across folds.

THE SENSITIVITY TEST
--------------------
days_until_membership_expiry is the feature most likely to encode the label's
own definition: the train_v2 population consists of subscribers whose
membership expires in March 2017, so the feature is bounded by construction
(observed max 31). Every model is therefore fitted twice, with and without it,
and the two are compared. If dropping it collapses performance, the signal was
definitional rather than behavioural and the stricter variant wins regardless
of which scores higher.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

log = logging.getLogger("train_real_v2")

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "processed" / "real_v2"
FEATURES_PARQUET = DATA_DIR / "features.parquet"
RESULTS_JSON = DATA_DIR / "model_results.json"
COMPARISON_JSON = DATA_DIR / "model_comparison.json"
PLOTS_DIR = DATA_DIR / "plots"

TARGET = "is_churn"
ID = "msno"
CATEGORICAL = ["city", "registered_via", "gender"]
EXPIRY_FEATURE = "days_until_membership_expiry"

SEED = 42
TEST_SIZE = 0.20
VAL_SIZE = 0.20          # of the full set, taken from the remaining 80%
DECISION_THRESHOLD = 0.50


def load() -> tuple[pd.DataFrame, pd.Series]:
    frame = pd.read_parquet(FEATURES_PARQUET)
    y = frame[TARGET].astype("int8")
    X = frame.drop(columns=[TARGET, ID])
    return X, y


def split(X: pd.DataFrame, y: pd.Series) -> dict:
    """Stratified 60/20/20. Stratification keeps the 8.99% positive rate
    stable across folds, which matters at this imbalance."""
    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y)
    val_fraction = VAL_SIZE / (1.0 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=val_fraction, random_state=SEED, stratify=y_rest)
    return {"train": (X_train, y_train), "val": (X_val, y_val),
            "test": (X_test, y_test)}


def _preprocessor(columns: list[str], scale: bool) -> ColumnTransformer:
    """Numeric imputation + optional scaling, one-hot for the three codes.

    Scaling is applied for Logistic Regression and skipped for the tree model,
    which is invariant to monotone rescaling. Imputation is median for the
    linear model; XGBoost is given the NaNs directly, since it learns a default
    direction per split and that is strictly more informative than a median.
    """
    numeric = [c for c in columns if c not in CATEGORICAL]
    steps: list = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        steps.append(("scale", StandardScaler()))
    return ColumnTransformer(
        [("num", Pipeline(steps), numeric),
         ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=0.001),
          CATEGORICAL)],
        verbose_feature_names_out=False)


def _preprocessor_tree(columns: list[str]) -> ColumnTransformer:
    numeric = [c for c in columns if c not in CATEGORICAL]
    return ColumnTransformer(
        [("num", "passthrough", numeric),
         ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=0.001),
          CATEGORICAL)],
        verbose_feature_names_out=False)


def evaluate(model, X, y, threshold: float = DECISION_THRESHOLD) -> dict:
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
        "accuracy": float(accuracy_score(y, pred)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "f1": float(f1_score(y, pred, zero_division=0)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp),
                             "fn": int(fn), "tp": int(tp)},
        "positive_rate_predicted": float(pred.mean()),
        "threshold": threshold,
    }


def build_models(columns: list[str]) -> dict:
    """Two families. Both get class weighting, because at a 9% positive rate an
    unweighted fit optimises accuracy by predicting almost no churn."""
    import xgboost as xgb

    return {
        "logistic_regression": Pipeline([
            ("preprocess", _preprocessor(columns, scale=True)),
            ("clf", LogisticRegression(max_iter=2000, solver="lbfgs",
                                       class_weight="balanced",
                                       random_state=SEED)),
        ]),
        "xgboost": Pipeline([
            ("preprocess", _preprocessor_tree(columns)),
            ("clf", xgb.XGBClassifier(
                n_estimators=400, max_depth=6, learning_rate=0.08,
                subsample=0.9, colsample_bytree=0.9, tree_method="hist",
                eval_metric="aucpr", random_state=SEED, n_jobs=-1,
                scale_pos_weight=float((1 - 0.0899) / 0.0899))),
        ]),
    }


def importances(name: str, model, columns: list[str], top: int = 15) -> list[dict]:
    """Ranked contributions, on each model's own terms.

    Logistic Regression: absolute standardised coefficient -- comparable across
    features because the inputs were scaled. XGBoost: gain.
    """
    pre = model.named_steps["preprocess"]
    clf = model.named_steps["clf"]
    try:
        names = list(pre.get_feature_names_out())
    except Exception:
        names = [f"f{i}" for i in range(len(columns))]

    if name == "logistic_regression":
        values = np.abs(clf.coef_[0])
        signed = clf.coef_[0]
    else:
        values = clf.feature_importances_
        signed = values

    order = np.argsort(-values)[:top]
    total = float(values.sum()) or 1.0
    return [{"feature": names[i], "importance": float(values[i]),
             "share": round(float(values[i]) / total, 6),
             "direction": ("increases risk" if signed[i] > 0 else "decreases risk")
             if name == "logistic_regression" else "n/a"}
            for i in order]


def run_variant(label: str, X: pd.DataFrame, y: pd.Series, drop: list[str]) -> dict:
    cols = [c for c in X.columns if c not in drop]
    Xv = X[cols]
    parts = split(Xv, y)
    X_train, y_train = parts["train"]
    X_val, y_val = parts["val"]
    X_test, y_test = parts["test"]

    out: dict = {"variant": label, "dropped_features": drop,
                 "feature_count": len(cols), "models": {}}
    for name, model in build_models(cols).items():
        t0 = time.time()
        model.fit(X_train, y_train)
        fit_seconds = time.time() - t0
        out["models"][name] = {
            "fit_seconds": round(fit_seconds, 1),
            "validation": evaluate(model, X_val, y_val),
            "test": evaluate(model, X_test, y_test),
            "top_features": importances(name, model, cols),
        }
        log.info("[%s] %s  val ROC %.4f  test ROC %.4f  (%.0fs)", label, name,
                 out["models"][name]["validation"]["roc_auc"],
                 out["models"][name]["test"]["roc_auc"], fit_seconds)
        out["models"][name]["_model"] = model
        out["models"][name]["_data"] = (X_test, y_test)
    return out


# ------------------------------------------------------------------ plots

def make_plots(variants: dict) -> list[str]:
    """ROC, precision-recall and confusion matrices for every model/variant."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    for curve in ("roc", "pr"):
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        for vlabel, variant in variants.items():
            for name, res in variant["models"].items():
                model = res["_model"]
                X_test, y_test = res["_data"]
                proba = model.predict_proba(X_test)[:, 1]
                if curve == "roc":
                    a, b, _ = roc_curve(y_test, proba)
                    score = res["test"]["roc_auc"]
                else:
                    b, a, _ = precision_recall_curve(y_test, proba)
                    score = res["test"]["pr_auc"]
                style = "-" if vlabel == "with_expiry" else "--"
                ax.plot(a, b, style, linewidth=1.6,
                        label=f"{name} [{vlabel}] {score:.4f}")
        if curve == "roc":
            ax.plot([0, 1], [0, 1], ":", color="#999", linewidth=1)
            ax.set_xlabel("False positive rate")
            ax.set_ylabel("True positive rate")
            ax.set_title("ROC - test set (KKBOX train_v2, cutoff 2017-02-28)")
        else:
            ax.axhline(0.0899, ls=":", color="#999", linewidth=1)
            ax.set_xlabel("Recall")
            ax.set_ylabel("Precision")
            ax.set_title("Precision-Recall - test set")
        ax.legend(fontsize=8, loc="lower right" if curve == "roc" else "upper right")
        ax.grid(alpha=0.25)
        path = PLOTS_DIR / f"{curve}_curve.png"
        fig.tight_layout()
        fig.savefig(path, dpi=140)
        plt.close(fig)
        written.append(str(path))

    pairs = [(v, n) for v in variants for n in variants[v]["models"]]
    fig, axes = plt.subplots(1, len(pairs), figsize=(3.4 * len(pairs), 3.4))
    for ax, (vlabel, name) in zip(np.atleast_1d(axes), pairs):
        cm = variants[vlabel]["models"][name]["test"]["confusion_matrix"]
        mat = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        ax.imshow(mat, cmap="Blues")
        for (i, j), v in np.ndenumerate(mat):
            ax.text(j, i, f"{v:,}", ha="center", va="center", fontsize=9,
                    color="white" if v > mat.max() / 2 else "black")
        ax.set_xticks([0, 1], ["pred 0", "pred 1"])
        ax.set_yticks([0, 1], ["true 0", "true 1"])
        ax.set_title(f"{name}\n[{vlabel}]", fontsize=9)
    fig.tight_layout()
    path = PLOTS_DIR / "confusion_matrices.png"
    fig.savefig(path, dpi=140)
    plt.close(fig)
    written.append(str(path))

    fig, ax = plt.subplots(figsize=(7.5, 4))
    rows = []
    for name in variants["with_expiry"]["models"]:
        for f in variants["with_expiry"]["models"][name]["top_features"]:
            if f["feature"] == EXPIRY_FEATURE:
                rows.append((name, f["share"]))
    if rows:
        ax.barh([r[0] for r in rows], [r[1] for r in rows], color="#2563eb")
        ax.set_xlabel(f"share of total importance held by {EXPIRY_FEATURE}")
        ax.set_title("Reliance on the expiry feature")
        ax.grid(alpha=0.25, axis="x")
        fig.tight_layout()
        path = PLOTS_DIR / "expiry_reliance.png"
        fig.savefig(path, dpi=140)
        plt.close(fig)
        written.append(str(path))
    else:
        plt.close(fig)
    return written


# ------------------------------------------------------------------ compare

def compare(variants: dict) -> dict:
    """Quantify what the expiry feature is worth, and judge whether that value
    reflects behaviour or the label definition."""
    a = variants["with_expiry"]["models"]
    b = variants["without_expiry"]["models"]
    per_model = {}
    for name in a:
        wa, wb = a[name]["test"], b[name]["test"]
        expiry_share = next(
            (f["share"] for f in a[name]["top_features"]
             if f["feature"] == EXPIRY_FEATURE), 0.0)
        expiry_rank = next(
            (i + 1 for i, f in enumerate(a[name]["top_features"])
             if f["feature"] == EXPIRY_FEATURE), None)
        per_model[name] = {
            "test_roc_auc_with": wa["roc_auc"],
            "test_roc_auc_without": wb["roc_auc"],
            "roc_auc_delta": round(wa["roc_auc"] - wb["roc_auc"], 6),
            "test_pr_auc_with": wa["pr_auc"],
            "test_pr_auc_without": wb["pr_auc"],
            "pr_auc_delta": round(wa["pr_auc"] - wb["pr_auc"], 6),
            "test_recall_with": wa["recall"],
            "test_recall_without": wb["recall"],
            "expiry_importance_share": expiry_share,
            "expiry_importance_rank": expiry_rank,
        }

    max_roc_delta = max(m["roc_auc_delta"] for m in per_model.values())
    max_share = max(m["expiry_importance_share"] for m in per_model.values())
    # Thresholds are judgement calls, stated openly rather than buried: one
    # feature worth more than 0.05 ROC-AUC, or holding over a third of all
    # importance, behaves like a definitional shortcut rather than a signal.
    suspicious = bool(max_roc_delta > 0.05 or max_share > 0.33)

    return {
        "per_model": per_model,
        "max_roc_auc_delta": round(max_roc_delta, 6),
        "max_expiry_importance_share": round(max_share, 6),
        "inflation_thresholds": {"roc_auc_delta": 0.05, "importance_share": 0.33},
        "suspicious_inflation_detected": suspicious,
        "primary_variant": "without_expiry" if suspicious else "with_expiry",
        "rationale": (
            "The expiry feature inflates performance beyond the stated "
            "thresholds, the signature of encoding the label definition rather "
            "than subscriber behaviour. The stricter variant is preferred."
            if suspicious else
            "Removing the expiry feature moves performance by less than the "
            "stated thresholds and the feature does not dominate importance, "
            "so it behaves as an ordinary predictor rather than a definitional "
            "shortcut."),
    }


def strip(variants: dict) -> dict:
    out = {}
    for vlabel, variant in variants.items():
        clean = {k: v for k, v in variant.items() if k != "models"}
        clean["models"] = {
            n: {k: v for k, v in res.items() if not k.startswith("_")}
            for n, res in variant["models"].items()}
        out[vlabel] = clean
    return out


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    t0 = time.time()
    X, y = load()
    parts = split(X, y)
    sizes = {k: {"rows": int(len(v[1])),
                 "positives": int(v[1].sum()),
                 "negatives": int((v[1] == 0).sum()),
                 "positive_rate": round(float(v[1].mean()), 6)}
             for k, v in parts.items()}
    log.info("split: %s", {k: v["rows"] for k, v in sizes.items()})

    variants = {
        "with_expiry": run_variant("with_expiry", X, y, drop=[]),
        "without_expiry": run_variant("without_expiry", X, y, drop=[EXPIRY_FEATURE]),
    }
    plots = make_plots(variants)
    comparison = compare(variants)

    results = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": {
            "path": str(FEATURES_PARQUET),
            "rows": int(len(X)),
            "feature_count": int(X.shape[1]),
            "observation_cutoff": "2017-02-28",
            "label_window": "March 2017 (train_v2)",
            "positive_rate": round(float(y.mean()), 6),
        },
        "split": {
            "strategy": "stratified random 60/20/20",
            "seed": SEED,
            "why_not_temporal": (
                "Every subscriber shares one observation cutoff (2017-02-28) "
                "and one label window (March 2017), so no time axis remains "
                "inside the dataset to hold out along. Leakage is controlled by "
                "the Stage 2 cutoff, not by the split."),
            "limitation": (
                "Measures generalisation to unseen subscribers in the same "
                "period, not to a future period."),
            "sizes": sizes,
        },
        "variants": strip(variants),
        "comparison": comparison,
        "plots": plots,
        "total_seconds": round(time.time() - t0, 1),
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(json.dumps(results, indent=2), encoding="utf-8")
    COMPARISON_JSON.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    log.info("wrote %s and %s", RESULTS_JSON, COMPARISON_JSON)

    print(json.dumps({"split": {k: v["rows"] for k, v in sizes.items()},
                      "comparison": comparison,
                      "total_seconds": results["total_seconds"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
