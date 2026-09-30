"""
Real-data (v2) prediction service.

Deliberately a separate module with separate state. The synthetic path in
main.py -- ChurnCastModel, the 27-feature table, /predict/{msno} -- is not
touched, imported into, or reconfigured by anything here. Either service can
fail to load without disabling the other, which is the point: the real model is
an addition, not a migration.

The two differ in more than their weights. The synthetic model scores 27
features built per-subscriber from generated data; this one scores 57 features
built from KKBOX transactions, members and user logs at a fixed 2017-02-28
cutoff. Mixing them would be meaningless, so the routes are kept apart and
every response carries data_mode so a caller can never confuse them.

TRAIN/SERVE ALIGNMENT
---------------------
The column contract written at registration time is re-read here and enforced
on every prediction: same columns, same order. The synthetic pipeline shipped
this exact bug twice -- a dtype round-trip and a bare read_csv that silently
changed code columns between fit and serve -- and both were invisible until a
registered model scored differently from its own artifact. Here a mismatch
raises instead.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd

from src import config
from src.api import recommendations

log = logging.getLogger("api.real")

MODEL_NAME = "ChurnCastRealV2"
DATA_MODE = "real_v2"
OBSERVATION_CUTOFF = "2017-02-28"

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "processed" / "real_v2"
FEATURES_PARQUET = DATA_DIR / "features.parquet"
FEATURE_COLUMNS_JSON = DATA_DIR / "model_feature_columns.json"

TARGET = "is_churn"
ID = "msno"

# Independent state. Nothing in main.py's _state is read or written.
state: dict[str, Any] = {
    "model": None,
    "model_version": None,
    "features": None,
    "feature_columns": None,
    "dtypes": None,
    "load_error": None,
}


def load_feature_columns() -> list[str]:
    """The column contract produced at registration time."""
    contract = json.loads(FEATURE_COLUMNS_JSON.read_text(encoding="utf-8"))
    state["dtypes"] = contract.get("dtypes", {})
    return list(contract["feature_columns"])


def load_features() -> pd.DataFrame:
    frame = pd.read_parquet(FEATURES_PARQUET)
    return frame.set_index(ID)


def load_model() -> None:
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions(f"name='{MODEL_NAME}'")
    if not versions:
        raise RuntimeError(
            f"No registered versions of '{MODEL_NAME}'. "
            "Run `python -m src.models.register_real_v2` first.")
    latest = max(versions, key=lambda v: int(v.version))
    state["model"] = mlflow.sklearn.load_model(f"models:/{MODEL_NAME}/{latest.version}")
    state["model_version"] = str(latest.version)


def startup() -> None:
    """Load everything the real route needs, recording failures rather than
    raising, so a missing real model cannot stop the synthetic API booting."""
    try:
        state["feature_columns"] = load_feature_columns()
        state["features"] = load_features()
        log.info("real feature table loaded: %s subscribers, %s columns",
                 f"{len(state['features']):,}", len(state["feature_columns"]))
    except Exception as e:
        state["load_error"] = f"real feature table: {e}"
        log.error("real feature table load failed: %s", e)
        return
    try:
        load_model()
        log.info("loaded %s v%s", MODEL_NAME, state["model_version"])
    except Exception as e:
        state["load_error"] = f"real model: {e}"
        log.error("real model load failed: %s", e)


def is_ready() -> bool:
    return state["model"] is not None and state["features"] is not None


def health() -> dict[str, Any]:
    return {
        "model_name": MODEL_NAME,
        "model_available": state["model"] is not None,
        "model_version": state["model_version"],
        "feature_table_available": state["features"] is not None,
        "subscribers": None if state["features"] is None else int(len(state["features"])),
        "feature_count": None if state["feature_columns"] is None
        else len(state["feature_columns"]),
        "observation_cutoff": OBSERVATION_CUTOFF,
        "data_mode": DATA_MODE,
        "detail": state["load_error"],
    }


def align(row: pd.DataFrame) -> pd.DataFrame:
    """Project one row onto the trained column contract, in order.

    Raises on a mismatch rather than letting a reordered or missing column
    through: silently wrong predictions are worse than a 500.
    """
    columns = state["feature_columns"]
    missing = [c for c in columns if c not in row.columns]
    if missing:
        raise RuntimeError(
            f"feature table is missing {len(missing)} trained column(s): "
            f"{missing[:5]}")
    return row.loc[:, columns]


def predict(msno: str) -> dict[str, Any]:
    """Score one subscriber with the registered real model.

    Raises KeyError when the subscriber is not in the table, so the route can
    turn that into a 404 rather than guessing.
    """
    features = state["features"]
    if msno not in features.index:
        raise KeyError(msno)

    row = features.loc[[msno]]
    row = row.drop(columns=[TARGET], errors="ignore")
    aligned = align(row)

    probability = float(state["model"].predict_proba(aligned)[0, 1])
    return {
        "msno": msno,
        "churn_probability": probability,
        "predicted_label": int(probability >= config.DECISION_THRESHOLD),
        "risk_level": config.risk_level(probability),
        "model_version": state["model_version"],
        "model_name": MODEL_NAME,
        "data_mode": DATA_MODE,
    }


# ------------------------------------------------------------------ analysis
#
# Signals for the real model come from XGBoost's own pred_contribs, which are
# exact SHAP values in log-odds space -- not a surrogate, not a permutation
# estimate. One-hot columns are summed back onto the feature that produced
# them, so "city" reads as one signal rather than twenty-two.

SIGNAL_LABELS: dict[str, str] = {
    "city": "City code",
    "registered_via": "Registration channel",
    "gender": "Gender",
    "age": "Age",
    "tenure_days": "Membership tenure",
    "registered_after_cutoff": "Registered after cutoff",
    "transaction_count": "Total transactions",
    "average_plan_days": "Average plan length",
    "average_list_price": "Average list price",
    "average_actual_paid": "Average amount paid",
    "total_paid": "Lifetime amount paid",
    "mean_discount": "Average discount",
    "auto_renew_rate": "Auto-renewal rate",
    "cancel_rate": "Cancellation rate",
    "cancel_count": "Prior cancellations",
    "txns_last_30d": "Transactions in last 30 days",
    "txns_last_90d": "Transactions in last 90 days",
    "cancels_last_30d": "Cancellations in last 30 days",
    "cancels_last_90d": "Cancellations in last 90 days",
    "distinct_payment_methods": "Distinct payment methods",
    "distinct_plan_days": "Distinct plan lengths",
    "days_since_last_transaction": "Days since last transaction",
    "transaction_history_days": "Transaction history span",
    "transaction_frequency": "Transaction frequency",
    "current_plan_days": "Current plan length",
    "current_price": "Current plan price",
    "current_actual_paid": "Current amount paid",
    "current_auto_renew": "Auto-renewal",
    "current_cancel_status": "Last transaction cancelled",
    "days_until_membership_expiry": "Days until membership expires",
    "log_active_days": "Total active listening days",
    "log_total_secs": "Total listening time",
    "log_average_daily_secs": "Average daily listening",
    "log_total_unique_songs": "Total unique songs",
    "log_average_unique_songs_per_active_day": "Unique songs per active day",
    "log_num_25": "Plays stopped before 25%",
    "log_num_50": "Plays stopped before 50%",
    "log_num_75": "Plays stopped before 75%",
    "log_num_985": "Plays reaching 98.5%",
    "log_num_100": "Completed plays",
    "log_completion_rate": "Play completion rate",
    "log_total_plays": "Total plays",
    "log_daily_secs_sd": "Daily listening variability",
    "log_active_days_last_7d": "Active days in last 7 days",
    "log_secs_last_7d": "Listening time in last 7 days",
    "log_activity_rate_last_7d": "Activity rate, last 7 days",
    "log_active_days_last_30d": "Active days in last 30 days",
    "log_secs_last_30d": "Listening time in last 30 days",
    "log_activity_rate_last_30d": "Activity rate, last 30 days",
    "log_active_days_last_90d": "Active days in last 90 days",
    "log_secs_last_90d": "Listening time in last 90 days",
    "log_activity_rate_last_90d": "Activity rate, last 90 days",
    "log_secs_trend_ratio": "Listening trend (30d vs prior 30d)",
    "log_secs_trend_delta": "Listening time change",
    "log_active_days_trend_delta": "Active days change",
    "log_days_since_last_listen": "Days since last listening",
    "log_listening_history_days": "Listening history span",
}

# Columns rendered as a coded identifier rather than a measurement.
CODE_FEATURES = {"city", "registered_via", "payment_method_last"}
BOOLEAN_FEATURES = {"current_auto_renew", "current_cancel_status",
                    "registered_after_cutoff"}
SECONDS_FEATURES = {"log_total_secs", "log_average_daily_secs", "log_secs_last_7d",
                    "log_secs_last_30d", "log_secs_last_90d", "log_secs_trend_delta",
                    "log_daily_secs_sd"}
RATIO_FEATURES = {"log_completion_rate", "cancel_rate", "auto_renew_rate",
                  "log_activity_rate_last_7d", "log_activity_rate_last_30d",
                  "log_activity_rate_last_90d"}


def format_value(feature: str, value: Any) -> str:
    """Render one feature value for display, honestly.

    A missing value says so rather than becoming a zero, because zero is a
    real measurement here and 'no data' is not.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "Not available"
    if feature in BOOLEAN_FEATURES:
        return "Enabled" if float(value) == 1 else "Disabled"
    if feature in CODE_FEATURES:
        return str(value)
    if feature == "gender":
        return str(value).capitalize() if str(value) != "unknown" else "Not available"
    if feature in SECONDS_FEATURES:
        hours = float(value) / 3600.0
        return f"{hours:,.1f} h" if abs(hours) >= 1 else f"{float(value):,.0f} s"
    if feature in RATIO_FEATURES:
        return f"{float(value) * 100:.1f}%"
    if isinstance(value, (int, float)):
        v = float(value)
        return f"{v:,.0f}" if abs(v) >= 100 or v == int(v) else f"{v:,.2f}"
    return str(value)


def _origin(transformed: str, columns: list[str]) -> str | None:
    """Map a post-encoding column back to the feature it came from."""
    if transformed in columns:
        return transformed
    for cat in ("city", "registered_via", "gender"):
        if transformed.startswith(f"{cat}_"):
            return cat
    return None


def explain(msno: str, top_n: int = 8) -> list[dict[str, Any]]:
    """Per-feature log-odds contributions for one subscriber, strongest first."""
    model = state["model"]
    columns = state["feature_columns"]
    row = state["features"].loc[[msno]].drop(columns=[TARGET], errors="ignore")
    aligned = align(row)

    try:
        import xgboost as xgb

        pre = model.named_steps["preprocess"]
        clf = model.named_steps["clf"]
        matrix = pre.transform(aligned)
        matrix = matrix.toarray() if hasattr(matrix, "toarray") else matrix
        names = list(pre.get_feature_names_out())

        booster = clf.get_booster()
        contribs = booster.predict(xgb.DMatrix(matrix, feature_names=names),
                                   pred_contribs=True)[0]
        # Last entry is the bias term, not a feature.
        per_feature: dict[str, float] = {}
        for name, value in zip(names, contribs[:-1]):
            origin = _origin(name, columns)
            if origin:
                per_feature[origin] = per_feature.get(origin, 0.0) + float(value)
    except Exception as e:                                        # pragma: no cover
        log.warning("real explain failed for %s: %s", msno, e)
        return []

    raw = aligned.iloc[0]
    ordered = sorted(per_feature.items(), key=lambda kv: -abs(kv[1]))
    out = []
    for feature, contribution in ordered:
        if abs(contribution) < 1e-6:
            continue
        out.append({
            "feature": feature,
            "label": SIGNAL_LABELS.get(feature, feature),
            "value": format_value(feature, raw.get(feature)),
            "contribution": round(contribution, 4),
            "direction": "increases" if contribution > 0 else "decreases",
        })
        if len(out) >= top_n:
            break
    return out


PROFILE_FIELDS = [
    ("City", "city", True),
    ("Registration channel", "registered_via", True),
    ("Gender", "gender", False),
    ("Age", "age", False),
    ("Membership tenure", "tenure_days", False),
    ("Auto-renewal", "current_auto_renew", False),
    ("Current plan length", "current_plan_days", False),
    ("Current plan price", "current_price", False),
]

BEHAVIOUR_FIELDS = [
    ("Active days (30d)", "log_active_days_last_30d"),
    ("Listening time (30d)", "log_secs_last_30d"),
    ("Unique songs per active day", "log_average_unique_songs_per_active_day"),
    ("Play completion rate", "log_completion_rate"),
    ("Days since last listening", "log_days_since_last_listen"),
    ("Listening trend (30d vs prior)", "log_secs_trend_ratio"),
    ("Transactions (30d)", "txns_last_30d"),
    ("Transaction frequency", "transaction_frequency"),
    ("Days since last transaction", "days_since_last_transaction"),
    ("Prior cancellations", "cancel_count"),
]


recommendations.register_labels(SIGNAL_LABELS)


def analysis(msno: str) -> dict[str, Any]:
    """Full real-mode subscriber view: prediction, snapshot, behaviour,
    model signals and the personalised retention plan."""
    features = state["features"]
    if msno not in features.index:
        raise KeyError(msno)

    prediction = predict(msno)
    probability = prediction["churn_probability"]
    row = features.loc[msno].drop(labels=[TARGET], errors="ignore")
    signals = explain(msno)

    profile = []
    for label, feature, is_code in PROFILE_FIELDS:
        if feature not in row.index:
            continue
        shown = format_value(feature, row.get(feature))
        if is_code and shown not in ("Not available", "unknown"):
            shown = f"Code {shown}"
        profile.append({"label": label, "feature": feature, "value": shown})

    behaviour = []
    for label, feature in BEHAVIOUR_FIELDS:
        if feature not in row.index:
            continue
        value = row.get(feature)
        behaviour.append({
            "label": label, "feature": feature,
            "value": format_value(feature, value),
            "raw": None if (value is None or pd.isna(value)) else float(value),
        })

    plan = recommendations.build_plan(probability, row, signals,
                                      formatter=format_value)

    return {
        "msno": msno,
        "data_mode": DATA_MODE,
        "model_name": MODEL_NAME,
        "observation_cutoff": OBSERVATION_CUTOFF,
        "prediction": prediction,
        "interpretation": (
            f"Based on the feature profile observed on or before "
            f"{OBSERVATION_CUTOFF}, the model estimates a "
            f"{probability * 100:.1f}% probability of churn in the label window."),
        "profile": profile,
        "behaviour": behaviour,
        "signals": signals,
        "signals_available": bool(signals),
        "recommendation_policy": {
            "severity": plan["severity"],
            "severity_label": plan["severity_label"],
            "risk_level": plan["risk_level"],
            "headline": plan["heading"],
            "intervention": plan["intervention"],
        },
        "plan": plan,
        "features": [
            {"feature": c, "label": SIGNAL_LABELS.get(c, c),
             "value": format_value(c, row.get(c))}
            for c in state["feature_columns"]
        ],
        "feature_count": len(state["feature_columns"]),
    }


REGISTRATION_JSON = DATA_DIR / "registration_real_v2.json"
MODEL_RESULTS_JSON = DATA_DIR / "model_results.json"


def model_details() -> dict[str, Any]:
    """Model card for the real model.

    Metrics are read from the artifacts the training and registration runs
    wrote; nothing is recomputed here and nothing is substituted when a file
    is absent -- the caller is told it is unavailable instead.
    """
    payload: dict[str, Any] = {
        "model_name": MODEL_NAME,
        "version": state["model_version"],
        "data_mode": DATA_MODE,
        "observation_cutoff": OBSERVATION_CUTOFF,
        "status": "READY" if state["model"] is not None else "UNAVAILABLE",
        "feature_count": None if state["feature_columns"] is None
        else len(state["feature_columns"]),
        "features": state["feature_columns"] or [],
        "subscribers": None if state["features"] is None else int(len(state["features"])),
        "tracking_uri": config.MLFLOW_TRACKING_URI,
        "decision_threshold": config.DECISION_THRESHOLD,
        "risk_thresholds": {"low_below": config.CHURN_RISK_LOW_THRESHOLD,
                            "high_at_or_above": config.CHURN_RISK_HIGH_THRESHOLD},
        "metrics_available": False,
    }
    try:
        reg = json.loads(REGISTRATION_JSON.read_text(encoding="utf-8"))
        payload.update({
            "metrics_available": True,
            "algorithm": reg["selected"]["family"],
            "variant": reg["selected"]["variant"],
            "experiment": reg["experiment"],
            "run_id": reg["run_id"],
            "validation": reg["validation"],
            "test": reg["test"],
            "top_features": reg["top_features"],
            "fit_seconds": reg["fit_seconds"],
        })
    except Exception as e:
        payload["detail"] = f"registration artifact unavailable: {e}"
    try:
        res = json.loads(MODEL_RESULTS_JSON.read_text(encoding="utf-8"))
        payload["split"] = res["split"]
        payload["dataset"] = res["dataset"]
        payload["expiry_sensitivity"] = res["comparison"]
    except Exception:
        pass
    return payload


# ------------------------------------------------- population analytics (batch)
#
# Served straight from the artifacts written by
# src/analytics/build_real_population.py. Nothing is scored per request: the
# model ran once offline over all 970,960 subscribers, and these endpoints are
# static reads plus filtering. Loaded lazily so a missing artifact degrades to
# a 503 on the dashboard rather than blocking API start-up.

POPULATION_PARQUET = DATA_DIR / "population_predictions.parquet"
POPULATION_JSON = DATA_DIR / "population_analytics.json"

population_state: dict[str, Any] = {
    "analytics": None,
    "predictions": None,
    "load_error": None,
    "loaded": False,
}


def load_population() -> None:
    """Read the batch artifacts once. Failures are recorded, not raised."""
    if population_state["loaded"]:
        return
    population_state["loaded"] = True
    try:
        population_state["analytics"] = json.loads(
            POPULATION_JSON.read_text(encoding="utf-8"))
        frame = pd.read_parquet(POPULATION_PARQUET)
        population_state["predictions"] = frame.set_index("msno")
        log.info("population artifacts loaded: %s scored subscribers",
                 f"{len(frame):,}")
    except Exception as e:
        population_state["load_error"] = f"population artifacts: {e}"
        log.warning("population artifacts unavailable: %s", e)


def population_ready() -> bool:
    load_population()
    return (population_state["analytics"] is not None
            and population_state["predictions"] is not None)


def population_analytics() -> dict[str, Any]:
    """Company-level analytics, exactly as the batch job computed them."""
    load_population()
    return population_state["analytics"]


def at_risk(limit: int = 12) -> list[dict[str, Any]]:
    """Highest-probability subscribers, with the model's own top risk signal.

    Sorted by probability so the table surfaces the population's actual worst
    cases rather than a sample.
    """
    load_population()
    frame = population_state["predictions"]
    top = frame.nlargest(limit, "churn_probability")
    return [_row(msno, r) for msno, r in top.iterrows()]


def _row(msno: str, r: pd.Series) -> dict[str, Any]:
    signal = r.get("top_signal")
    return {
        "msno": str(msno),
        "churn_probability": float(r["churn_probability"]),
        "predicted_label": int(r["predicted_label"]),
        "risk_level": str(r["risk_level"]),
        "severity": str(r.get("severity", "")),
        "top_signal": None if signal is None or pd.isna(signal) else str(signal),
        "top_signal_label": (SIGNAL_LABELS.get(str(signal), str(signal))
                             if signal is not None and not pd.isna(signal) else None),
        "top_signal_contribution": (
            None if pd.isna(r.get("top_signal_contribution"))
            else round(float(r["top_signal_contribution"]), 4)),
        "activity_band": str(r.get("activity_band", "")),
        "active_days_30d": (None if pd.isna(r.get("active_days_30d"))
                            else int(r["active_days_30d"])),
        "auto_renew": (None if pd.isna(r.get("auto_renew"))
                       else int(r["auto_renew"])),
        "gender": str(r.get("gender", "unknown")),
    }


def browse(query: str = "", risk: str = "", auto_renew: str = "",
           activity: str = "", min_probability: float | None = None,
           max_probability: float | None = None,
           limit: int = 25, offset: int = 0,
           sort: str = "probability_desc") -> dict[str, Any]:
    """Filtered, paginated view over the scored population."""
    load_population()
    frame = population_state["predictions"]

    if query:
        frame = frame[frame.index.astype(str).str.contains(
            query.strip(), case=False, na=False, regex=False)]
    if risk:
        frame = frame[frame["risk_level"] == risk.upper()]
    if activity:
        frame = frame[frame["activity_band"] == activity]
    if auto_renew == "1":
        frame = frame[frame["auto_renew"] == 1]
    elif auto_renew == "0":
        frame = frame[frame["auto_renew"] == 0]
    elif auto_renew == "unknown":
        frame = frame[frame["auto_renew"].isna()]
    if min_probability is not None:
        frame = frame[frame["churn_probability"] >= min_probability]
    if max_probability is not None:
        frame = frame[frame["churn_probability"] <= max_probability]

    total = int(len(frame))
    if sort == "probability_asc":
        frame = frame.sort_values("churn_probability")
    elif sort == "probability_desc":
        frame = frame.sort_values("churn_probability", ascending=False)

    page = frame.iloc[offset: offset + limit]
    return {
        "total": total, "limit": limit, "offset": offset,
        "data_mode": DATA_MODE,
        "rows": [_row(msno, r) for msno, r in page.iterrows()],
    }


def population_filters() -> dict[str, Any]:
    """Filter values that actually occur in the scored population."""
    load_population()
    frame = population_state["predictions"]
    bands = [b for b in frame["activity_band"].dropna().unique().tolist()]
    order = ["High activity", "Moderate activity", "Low activity", "Inactive"]
    return {
        "risk_levels": ["LOW", "MEDIUM", "HIGH"],
        "activity_bands": [b for b in order if b in bands],
        "auto_renew": [{"value": "1", "label": "Enabled"},
                       {"value": "0", "label": "Disabled"},
                       {"value": "unknown", "label": "Unknown"}],
        "data_mode": DATA_MODE,
    }
