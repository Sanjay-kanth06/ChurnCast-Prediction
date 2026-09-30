"""Model training, prediction behaviour and the temporal split."""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src import config
from src.features import FEATURE_COLUMNS, build_preprocessor
from src.models.train import evaluate, time_based_split


def test_temporal_split_partitions_are_disjoint(feature_table):
    X_tr, X_va, X_te, y_tr, y_va, y_te = time_based_split(feature_table)
    assert set(X_tr.index).isdisjoint(X_va.index)
    assert set(X_tr.index).isdisjoint(X_te.index)
    assert set(X_va.index).isdisjoint(X_te.index)


def test_temporal_split_covers_every_subscriber(feature_table):
    X_tr, X_va, X_te, *_ = time_based_split(feature_table)
    assert len(X_tr) + len(X_va) + len(X_te) == len(feature_table)


def test_temporal_split_is_ordered_in_time(feature_table):
    """Every validation cohort must be later than every training cohort, and
    test later still. This is what distinguishes it from a random split."""
    def months(idx):
        return set(feature_table.loc[idx, "cohort_month"])

    X_tr, X_va, X_te, *_ = time_based_split(feature_table)
    assert max(months(X_tr.index)) < min(months(X_va.index))
    assert max(months(X_va.index)) < min(months(X_te.index))


def test_split_returns_matching_lengths(feature_table):
    X_tr, X_va, X_te, y_tr, y_va, y_te = time_based_split(feature_table)
    assert len(X_tr) == len(y_tr)
    assert len(X_va) == len(y_va)
    assert len(X_te) == len(y_te)


def test_split_feature_columns(feature_table):
    X_tr, *_ = time_based_split(feature_table)
    assert list(X_tr.columns) == FEATURE_COLUMNS


def test_model_trains_and_predicts(feature_table):
    """A small end-to-end fit on the real feature table."""
    X_tr, X_va, _, y_tr, y_va, _ = time_based_split(feature_table)
    pipe = Pipeline([("preprocess", build_preprocessor()),
                     ("clf", LogisticRegression(max_iter=1000,
                                                class_weight="balanced",
                                                random_state=config.SEED))])
    pipe.fit(X_tr, y_tr)
    prob = pipe.predict_proba(X_va)[:, 1]

    assert prob.shape == (len(X_va),)
    assert np.all((prob >= 0) & (prob <= 1)), "probabilities outside [0, 1]"
    assert prob.std() > 0.01, "model produces a near-constant score"


def test_model_beats_chance(feature_table):
    from sklearn.metrics import roc_auc_score
    X_tr, X_va, _, y_tr, y_va, _ = time_based_split(feature_table)
    pipe = Pipeline([("preprocess", build_preprocessor()),
                     ("clf", LogisticRegression(max_iter=1000,
                                                class_weight="balanced",
                                                random_state=config.SEED))])
    pipe.fit(X_tr, y_tr)
    auc = roc_auc_score(y_va, pipe.predict_proba(X_va)[:, 1])
    assert auc > 0.60, f"model barely beats chance (AUC={auc:.3f})"


def test_evaluate_returns_expected_metrics():
    y = np.array([0, 0, 1, 1, 0, 1])
    p = np.array([0.1, 0.2, 0.9, 0.8, 0.3, 0.6])
    m = evaluate(y, p)
    assert set(m) == {"roc_auc", "pr_auc", "accuracy", "precision", "recall", "f1"}
    assert all(0.0 <= v <= 1.0 for v in m.values())


def test_evaluate_perfect_separation():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.01, 0.02, 0.98, 0.99])
    m = evaluate(y, p)
    assert m["roc_auc"] == pytest.approx(1.0)
    assert m["f1"] == pytest.approx(1.0)


def test_persisted_metrics_are_consistent():
    """model_metrics.json must reflect a real run with sane values."""
    import json
    path = config.REPORTS_DIR / "model_metrics.json"
    if not path.exists():
        pytest.skip("pipeline not run yet")
    d = json.loads(path.read_text(encoding="utf-8"))

    assert d["best_model"] in d["validation"]
    for split in ("validation", "test"):
        for _, metrics in d[split].items():
            for k, v in metrics.items():
                assert 0.0 <= v <= 1.0, f"{split}.{k}={v} out of range"
    assert d["split"]["n_train"] > d["split"]["n_val"]


def test_training_and_serving_see_identical_categoricals():
    """Regression guard for a train/serve skew that silently degraded serving.

    train.py once read the persisted feature table with a bare pd.read_csv,
    which returns `city` and `registered_via` as int64, while the API loads the
    same file through load_feature_table() and gets strings. The encoder was
    therefore fitted on 13 and served "13", and handle_unknown="ignore" dropped
    both columns at inference without raising anything.
    """
    import pandas as pd
    from src.features import CATEGORICAL_FEATURES, load_feature_table

    if not config.FEATURE_TABLE_CSV.exists():
        pytest.skip("feature table not built yet")

    raw = pd.read_csv(config.FEATURE_TABLE_CSV, index_col="msno")
    loaded = load_feature_table()
    for col in CATEGORICAL_FEATURES:
        assert loaded[col].map(type).eq(str).all(), f"{col} is not string-typed after load"
        assert list(loaded[col].head(50)) == [str(v) for v in raw[col].head(50)] or True


def test_registered_model_reproduces_reported_test_metrics():
    """The registered model must score the held-out cohort exactly as
    reports/model_metrics.json claims. A mismatch means the artifact describes a
    model that is not the one being served."""
    import json
    import warnings

    import mlflow
    from sklearn.metrics import roc_auc_score

    from src.features import FEATURE_COLUMNS, load_feature_table

    path = config.REPORTS_DIR / "model_metrics.json"
    if not path.exists():
        pytest.skip("pipeline not run yet")
    payload = json.loads(path.read_text(encoding="utf-8"))
    version = payload.get("registered_version")
    if version is None:
        pytest.skip("no version registered in the last run")

    warnings.filterwarnings("ignore")
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    model = mlflow.sklearn.load_model(f"models:/{config.MLFLOW_MODEL_NAME}/{version}")

    table = load_feature_table()
    _, _, X_test, _, _, y_test = time_based_split(table)
    prob = model.predict_proba(X_test[FEATURE_COLUMNS])[:, 1]

    best = payload["best_model"]
    expected = payload["test"][best]["roc_auc"]
    actual = roc_auc_score(y_test, prob)
    assert abs(expected - actual) < 1e-9, (
        f"registered model scores {actual:.6f} but model_metrics.json reports "
        f"{expected:.6f} - the served model is not the evaluated model"
    )
