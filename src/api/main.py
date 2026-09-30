"""
ChurnCast prediction API.

Endpoints
---------
GET  /health                     service and model availability
GET  /model-info                 registered model metadata and risk thresholds
GET  /subscribers/sample         a few valid subscriber IDs for the demo UI
GET  /subscriber/{msno}/features the stored feature vector for one subscriber
POST /predict                    predict from supplied feature values
POST /predict/{msno}             PRIMARY: predict by subscriber ID (feature lookup)

The model is resolved from the MLflow Model Registry at start-up. A failure to
load is recorded rather than raised, so /health can report the condition instead
of the process dying.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, Optional

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException, Path as PathParam, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src import config
from src.api import analytics
from src.api import real_service
from src.api import recommendations
from src.features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("api")

_state: dict[str, Any] = {
    "model": None, "model_version": None, "model_type": None,
    "features": None, "defaults": None, "load_error": None,
    # derived once at start-up so browse and dashboard views do not re-score
    "scored": None, "attribution": None, "metrics": None,
}


# ------------------------------------------------------------------ loading

def _load_model() -> None:
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    versions = client.search_model_versions(f"name='{config.MLFLOW_MODEL_NAME}'")
    if not versions:
        raise RuntimeError(
            f"No registered versions of '{config.MLFLOW_MODEL_NAME}'. "
            "Run `python scripts/run_pipeline.py` first."
        )
    latest = max(versions, key=lambda v: int(v.version))
    _state["model"] = mlflow.sklearn.load_model(f"models:/{config.MLFLOW_MODEL_NAME}/{latest.version}")
    _state["model_version"] = str(latest.version)
    try:
        _state["model_type"] = client.get_run(latest.run_id).data.tags.get("model_type")
    except Exception:
        _state["model_type"] = None
    log.info("loaded %s v%s (%s)", config.MLFLOW_MODEL_NAME,
             latest.version, _state["model_type"])


def _load_features() -> None:
    from src.features import load_feature_table
    ft = load_feature_table()
    _state["features"] = ft
    # Defaults used to complete a partial /predict payload, so the model always
    # receives the exact column set it was trained on.
    defaults = {}
    for c in NUMERIC_FEATURES:
        defaults[c] = float(ft[c].median())
    for c in CATEGORICAL_FEATURES:
        mode = ft[c].mode()
        defaults[c] = str(mode.iloc[0]) if len(mode) else "unknown"
    _state["defaults"] = defaults
    log.info("feature table loaded: %d subscribers", len(ft))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Real (v2) service: loaded independently. A failure here is recorded in
    # its own state and leaves the synthetic path untouched.
    try:
        real_service.startup()
    except Exception as e:                                        # pragma: no cover
        log.error("real service startup failed: %s", e)
    try:
        _load_features()
    except Exception as e:
        _state["load_error"] = f"feature table: {e}"
        log.error("feature table load failed: %s", e)
    try:
        _load_model()
    except Exception as e:
        _state["load_error"] = f"model: {e}"
        log.error("model load failed: %s", e)

    # Derived views. Each failure is logged and leaves its slot None; the
    # corresponding endpoint then reports unavailability rather than inventing
    # a value.
    _state["metrics"] = analytics.load_model_metrics()
    if _state["model"] is not None and _state["features"] is not None:
        try:
            _state["scored"] = analytics.score_all(_state["model"], _state["features"])
        except Exception as e:
            log.error("batch scoring failed: %s", e)
        try:
            _state["attribution"] = analytics.build_attribution(_state["model"])
        except Exception as e:
            log.warning("attribution unavailable: %s", e)
    yield


app = FastAPI(
    title="ChurnCast API",
    description="Subscriber churn prediction — PROTOTYPE running on KKBOX-like synthetic data.",
    version="2.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------------ schemas

class PredictRequest(BaseModel):
    """Feature-based prediction. Any feature omitted is filled from the training
    median (numeric) or mode (categorical), so the model always receives its
    full trained column set."""
    msno: str = Field(..., examples=["SYN000001"])
    recency_days: Optional[float] = Field(None, ge=0)
    txns_last_30d: Optional[float] = Field(None, ge=0)
    txns_last_90d: Optional[float] = Field(None, ge=0)
    active_days_last_30d: Optional[float] = Field(None, ge=0)
    active_days_last_90d: Optional[float] = Field(None, ge=0)
    total_secs_last_30d: Optional[float] = Field(None, ge=0)
    total_secs_last_90d: Optional[float] = Field(None, ge=0)
    engagement_trend: Optional[float] = None
    recent_activity_ratio: Optional[float] = Field(None, ge=0)
    skip_ratio: Optional[float] = Field(None, ge=0, le=1)
    completion_ratio: Optional[float] = Field(None, ge=0, le=1)
    plan_price: Optional[float] = Field(None, ge=0)
    payment_amount: Optional[float] = Field(None, ge=0)
    is_auto_renew: Optional[int] = Field(None, ge=0, le=1)
    age: Optional[float] = Field(None, ge=0, le=120)
    tenure_days: Optional[float] = Field(None, ge=0)

    model_config = {"json_schema_extra": {"example": {
        "msno": "SYN000001", "recency_days": 5, "txns_last_90d": 4,
        "active_days_last_30d": 18, "engagement_trend": 0.82,
        "plan_price": 149, "is_auto_renew": 1, "age": 24, "tenure_days": 540,
    }}}


class PredictResponse(BaseModel):
    msno: str
    churn_probability: float = Field(..., ge=0, le=1)
    predicted_label: int = Field(..., ge=0, le=1)
    risk_level: str
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_available: bool
    model_version: Optional[str] = None
    feature_table_available: bool
    subscribers: Optional[int] = None
    detail: Optional[str] = None


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: Optional[str]
    model_type: Optional[str]
    mlflow_tracking_uri: str
    mlflow_connected: bool
    feature_count: int
    features: list[str]
    subscribers: Optional[int]
    dataset: str
    risk_thresholds: dict
    decision_threshold: float


class SubscriberFeatures(BaseModel):
    msno: str
    features: dict
    is_churn: Optional[int] = None


def _validate_msno(value: str) -> str:
    v = (value or "").strip()
    if not v:
        raise HTTPException(status_code=422, detail="Subscriber ID must not be empty.")
    if len(v) > 64:
        raise HTTPException(status_code=422, detail="Subscriber ID is too long.")
    return v


def _require_ready() -> None:
    if _state["model"] is None:
        raise HTTPException(status_code=503,
                            detail="Model not loaded. Check /health.")
    if _state["features"] is None:
        raise HTTPException(status_code=503,
                            detail="Feature table not loaded. Check /health.")


def _predict_frame(row: pd.DataFrame, msno: str) -> PredictResponse:
    prob = float(_state["model"].predict_proba(row[FEATURE_COLUMNS])[0, 1])
    return PredictResponse(
        msno=msno,
        churn_probability=round(prob, 4),
        predicted_label=int(prob >= config.DECISION_THRESHOLD),
        risk_level=config.risk_level(prob),
        model_version=str(_state["model_version"]),
    )


# ------------------------------------------------------------------ endpoints

@app.get("/health", response_model=HealthResponse)
def health():
    ok = _state["model"] is not None and _state["features"] is not None
    return HealthResponse(
        status="ok" if ok else "degraded",
        model_available=_state["model"] is not None,
        model_version=_state["model_version"],
        feature_table_available=_state["features"] is not None,
        subscribers=None if _state["features"] is None else int(len(_state["features"])),
        detail=_state["load_error"],
    )


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    return ModelInfoResponse(
        model_name=config.MLFLOW_MODEL_NAME,
        model_version=_state["model_version"],
        model_type=_state["model_type"],
        mlflow_tracking_uri=config.MLFLOW_TRACKING_URI,
        mlflow_connected=_state["model"] is not None,
        feature_count=len(FEATURE_COLUMNS),
        features=FEATURE_COLUMNS,
        subscribers=None if _state["features"] is None else int(len(_state["features"])),
        dataset="KKBOX-like synthetic data (prototype)",
        risk_thresholds={"low_below": config.CHURN_RISK_LOW_THRESHOLD,
                         "high_at_or_above": config.CHURN_RISK_HIGH_THRESHOLD},
        decision_threshold=config.DECISION_THRESHOLD,
    )


@app.get("/subscribers/sample")
def sample_subscribers(n: int = 5):
    if _state["features"] is None:
        raise HTTPException(status_code=503, detail="Feature table not loaded.")
    n = max(1, min(int(n), 25))
    return {"subscribers": list(_state["features"].index[:n])}


@app.get("/subscriber/{msno}/features", response_model=SubscriberFeatures)
def subscriber_features(msno: str = PathParam(..., examples=["SYN000001"])):
    msno = _validate_msno(msno)
    if _state["features"] is None:
        raise HTTPException(status_code=503, detail="Feature table not loaded.")
    if msno not in _state["features"].index:
        raise HTTPException(status_code=404, detail=f"Subscriber {msno} not found")
    row = _state["features"].loc[msno]
    feats = {c: (None if pd.isna(row[c]) else
                 (float(row[c]) if c in NUMERIC_FEATURES else str(row[c])))
             for c in FEATURE_COLUMNS}
    churn = row.get("is_churn")
    return SubscriberFeatures(msno=msno, features=feats,
                              is_churn=None if pd.isna(churn) else int(churn))


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    """Predict from supplied feature values. Intended for testing and API
    integration; the user-facing path is POST /predict/{msno}."""
    _require_ready()
    msno = _validate_msno(req.msno)

    values = dict(_state["defaults"])
    supplied = req.model_dump(exclude_none=True)
    supplied.pop("msno", None)
    for k, v in supplied.items():
        if k in values:
            values[k] = str(v) if k in CATEGORICAL_FEATURES else float(v)

    return _predict_frame(pd.DataFrame([values]), msno)


@app.post("/predict/{msno}", response_model=PredictResponse)
def predict_by_msno(msno: str = PathParam(..., examples=["SYN000001"])):
    """PRIMARY user-facing endpoint: looks the subscriber's engineered features
    up in the feature table and scores them with the registered model."""
    msno = _validate_msno(msno)
    _require_ready()

    ft = _state["features"]
    if msno not in ft.index:
        raise HTTPException(status_code=404, detail=f"Subscriber {msno} not found")

    row = ft.loc[[msno]].copy()
    for c in CATEGORICAL_FEATURES:
        row[c] = row[c].astype(str)
    return _predict_frame(row, msno)


# ------------------------------------------------------------------ UI endpoints
#
# Added for the ChurnCast web interface. Every value returned here is derived
# from the feature table, the registered model or reports/model_metrics.json.
# Where something does not exist in the data it is omitted or reported as
# unavailable rather than substituted.

def _require_scored() -> pd.DataFrame:
    if _state["scored"] is None:
        raise HTTPException(status_code=503,
                            detail="Subscriber scores unavailable. Check /health.")
    return _state["scored"]


@app.get("/dashboard/summary")
def dashboard_summary():
    """Project-level overview. The risk distribution is computed by scoring the
    whole subscriber population with the registered model, not estimated."""
    _require_ready()
    scored = _require_scored()
    ft = _state["features"]
    metrics = _state["metrics"] or {}
    best = metrics.get("best_model")
    val = (metrics.get("validation") or {}).get(best, {}) if best else {}
    test = (metrics.get("test") or {}).get(best, {}) if best else {}

    return {
        "total_subscribers": int(len(ft)),
        "feature_count": len(FEATURE_COLUMNS),
        "model": {
            "name": config.MLFLOW_MODEL_NAME,
            "version": _state["model_version"],
            "algorithm": _state["model_type"],
            "status": "READY",
            "validation_roc_auc": val.get("roc_auc"),
            "test_roc_auc": test.get("roc_auc"),
        },
        "risk_distribution": analytics.risk_distribution(scored),
        "risk_thresholds": {"low_below": config.CHURN_RISK_LOW_THRESHOLD,
                            "high_at_or_above": config.CHURN_RISK_HIGH_THRESHOLD},
        "dataset": "KKBOX-like synthetic data (prototype)",
        "cohorts": sorted(ft["cohort_month"].unique().tolist())
        if "cohort_month" in ft.columns else [],
        "system_status": "ONLINE",
    }


@app.get("/filters")
def filters():
    """Filter values that actually occur in the feature table."""
    _require_ready()
    ft = _state["features"]

    def values(col: str) -> list[str]:
        if col not in ft.columns:
            return []
        return sorted(v for v in ft[col].astype(str).unique() if v != "unknown")

    return {
        "risk_levels": ["LOW", "MEDIUM", "HIGH"],
        "auto_renew": [{"value": "1", "label": "Enabled"},
                       {"value": "0", "label": "Disabled"}],
        "cities": values("city"),
        "registered_via": values("registered_via"),
    }


@app.get("/subscribers")
def list_subscribers(
    query: str = Query("", description="Subscriber ID substring match"),
    risk: str = Query("", description="LOW | MEDIUM | HIGH"),
    auto_renew: str = Query("", description="1 | 0"),
    city: str = Query(""),
    limit: int = Query(25, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """Paginated subscriber browse view, joined to the cached predictions."""
    _require_ready()
    scored = _require_scored()
    ft = _state["features"]

    df = ft.join(scored, how="inner")
    if query:
        df = df[df.index.astype(str).str.contains(query.strip(), case=False, na=False)]
    if risk:
        df = df[df["risk_level"] == risk.upper()]
    if auto_renew:
        df = df[df["is_auto_renew"].astype(str) == auto_renew]
    if city:
        df = df[df["city"].astype(str) == city]

    total = int(len(df))
    page = df.iloc[offset: offset + limit]

    rows = []
    for msno, r in page.iterrows():
        rows.append({
            "msno": str(msno),
            "city": analytics.format_value("city", r.get("city")),
            "registered_via": analytics.format_value("registered_via", r.get("registered_via")),
            "auto_renew": analytics.format_value("is_auto_renew", r.get("is_auto_renew")),
            "tenure_days": None if pd.isna(r.get("tenure_days")) else int(r["tenure_days"]),
            "active_days_last_30d": None if pd.isna(r.get("active_days_last_30d"))
            else int(r["active_days_last_30d"]),
            "churn_probability": float(r["churn_probability"]),
            "predicted_label": int(r["predicted_label"]),
            "risk_level": str(r["risk_level"]),
        })

    return {"total": total, "limit": limit, "offset": offset, "rows": rows}


@app.get("/subscriber/{msno}/analysis")
def subscriber_analysis(msno: str = PathParam(..., examples=["SYN000001"])):
    """Full subscriber view: profile, prediction, model signals and suggested
    retention actions. Signals are the registered model's own log-odds
    contributions; recommendations are deterministic rules over real feature
    values."""
    msno = _validate_msno(msno)
    _require_ready()
    ft = _state["features"]
    if msno not in ft.index:
        raise HTTPException(status_code=404, detail=f"Subscriber {msno} not found")

    row_df = ft.loc[[msno]].copy()
    for c in CATEGORICAL_FEATURES:
        row_df[c] = row_df[c].astype(str)
    row = row_df.iloc[0]

    prediction = _predict_frame(row_df, msno)
    prob = prediction.churn_probability
    # Computed once: the recommendation ranking reuses the model's own
    # attribution so the strongest real evidence leads.
    signals = analytics.explain(_state["attribution"], row_df)

    feature_values = []
    for c in FEATURE_COLUMNS:
        raw = row.get(c)
        missing = raw is None or (isinstance(raw, float) and pd.isna(raw))
        feature_values.append({
            "feature": c,
            "label": analytics.SIGNAL_LABELS.get(c, c),
            "value": analytics.format_value(c, raw),
            "raw": None if missing else (float(raw) if c in NUMERIC_FEATURES else str(raw)),
        })

    return {
        "msno": msno,
        "prediction": prediction.model_dump(),
        "interpretation": analytics.interpretation(prob),
        "profile": analytics.profile(row),
        "signals": signals,
        "signals_available": _state["attribution"] is not None,
        "recommendation_policy": analytics.recommendation_policy(prob),
        "recommendations": analytics.recommend(prob, row, signals),
        # Personalised plan from the shared rule engine. Additive: the two
        # keys above are unchanged so existing consumers keep working.
        "plan": recommendations.build_plan(
            prob, row, signals, formatter=analytics.format_value),
        "features": feature_values,
        "feature_count": len(FEATURE_COLUMNS),
    }


@app.get("/model-details")
def model_details():
    """Full model card: verified metrics from the training run, pipeline stages,
    registry state and the promotion rule. Metrics come from
    reports/model_metrics.json and are reported only if that file exists."""
    _require_ready()
    metrics = _state["metrics"]
    payload: dict[str, Any] = {
        "model_name": config.MLFLOW_MODEL_NAME,
        "version": _state["model_version"],
        "algorithm": _state["model_type"],
        "status": "READY",
        "feature_count": len(FEATURE_COLUMNS),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "mlflow_tracking_uri": config.MLFLOW_TRACKING_URI,
        "dataset": "KKBOX-like synthetic data (prototype)",
        "decision_threshold": config.DECISION_THRESHOLD,
        "risk_thresholds": {"low_below": config.CHURN_RISK_LOW_THRESHOLD,
                            "high_at_or_above": config.CHURN_RISK_HIGH_THRESHOLD},
        "metrics_available": metrics is not None,
        "pipeline": ["Raw data", "Data validation", "Feature engineering",
                     "Temporal split", "Model training", "Evaluation",
                     "MLflow tracking", "Model registry", "FastAPI", "Prediction"],
        "dag": {"dag_id": "churncast_retrain",
                "tasks": ["ingest", "build_features", "train", "evaluate",
                          "register_if_better"],
                "schedule": "@weekly"},
        "promotion_rule": ("A newly trained model replaces the registered one only "
                           "when its validation ROC-AUC strictly exceeds the "
                           "incumbent's."),
    }
    if metrics:
        payload.update({
            "best_model": metrics.get("best_model"),
            "seed": metrics.get("seed"),
            "n_subscribers": metrics.get("n_subscribers"),
            "split": metrics.get("split"),
            "validation": metrics.get("validation"),
            "test": metrics.get("test"),
            "registry_decision": metrics.get("registry_decision"),
        })
    return payload


# ------------------------------------------------------- real (v2) endpoints
#
# Separate model, separate feature table, separate state. The synthetic routes
# above are unchanged; these are additive. Every response carries data_mode so
# a caller cannot confuse a real prediction with a synthetic one.

class RealPredictResponse(BaseModel):
    msno: str
    churn_probability: float = Field(..., ge=0, le=1)
    predicted_label: int = Field(..., ge=0, le=1)
    risk_level: str
    model_version: Optional[str] = None
    model_name: str
    data_mode: str


def _require_real_ready() -> None:
    if not real_service.is_ready():
        raise HTTPException(
            status_code=503,
            detail=(real_service.state["load_error"]
                    or "Real model unavailable. Check /health/real."))


@app.get("/health/real")
def health_real():
    """Health of the real service alone, so it can be diagnosed separately."""
    return real_service.health()


@app.post("/predict-real/{msno}", response_model=RealPredictResponse)
def predict_real(msno: str = PathParam(..., examples=["SYN000001"])):
    """Predict churn for one real KKBOX subscriber.

    msno -> real_v2 feature table -> trained column contract -> ChurnCastRealV2
    from the MLflow registry -> probability, label, risk band.
    """
    msno = _validate_msno(msno)
    _require_real_ready()
    try:
        return real_service.predict(msno)
    except KeyError:
        raise HTTPException(status_code=404,
                            detail=f"Subscriber {msno} not found in the real dataset")
    except RuntimeError as e:
        # Train/serve column mismatch: fail loudly rather than score wrongly.
        log.error("real prediction failed for %s: %s", msno, e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/subscribers-real/sample")
def sample_real_subscribers(n: int = 5):
    """A few valid real subscriber IDs, for smoke tests and the UI."""
    _require_real_ready()
    n = max(1, min(int(n), 25))
    return {"subscribers": list(real_service.state["features"].index[:n]),
            "data_mode": real_service.DATA_MODE}


@app.get("/subscriber-real/{msno}/analysis")
def subscriber_real_analysis(msno: str = PathParam(..., examples=["msno_hash="])):
    """Full real-mode subscriber view.

    Prediction, customer snapshot, behavioural metrics, the model's own
    log-odds contributions and a personalised retention plan -- all derived
    from this subscriber's stored feature row and the registered
    ChurnCastRealV2 model.
    """
    msno = _validate_msno(msno)
    _require_real_ready()
    try:
        return real_service.analysis(msno)
    except KeyError:
        raise HTTPException(status_code=404,
                            detail=f"Subscriber {msno} not found in the real dataset")
    except RuntimeError as e:
        log.error("real analysis failed for %s: %s", msno, e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/model-details/real")
def model_details_real():
    """Model card for ChurnCastRealV2, read from the training artifacts."""
    _require_real_ready()
    return real_service.model_details()


# ------------------------------------------- real population analytics (batch)
#
# Static reads over the artifacts produced by the offline scoring job. Nothing
# here scores a subscriber; the whole population was scored once, offline.

def _require_population() -> None:
    if not real_service.population_ready():
        raise HTTPException(
            status_code=503,
            detail=(real_service.population_state["load_error"]
                    or "Population analytics not built. Run "
                       "`python -m src.analytics.build_real_population`."))


@app.get("/population/analytics")
def population_analytics():
    """Company-level analytics for the real subscriber population."""
    _require_population()
    return real_service.population_analytics()


@app.get("/population/at-risk")
def population_at_risk(limit: int = Query(12, ge=1, le=100)):
    """Highest-probability subscribers, with the model's own top risk signal."""
    _require_population()
    return {"rows": real_service.at_risk(limit),
            "data_mode": real_service.DATA_MODE}


@app.get("/population/filters")
def population_filters():
    """Filter values present in the scored population."""
    _require_population()
    return real_service.population_filters()


@app.get("/population/subscribers")
def population_subscribers(
    query: str = Query(""),
    risk: str = Query(""),
    auto_renew: str = Query(""),
    activity: str = Query(""),
    min_probability: Optional[float] = Query(None, ge=0, le=1),
    max_probability: Optional[float] = Query(None, ge=0, le=1),
    sort: str = Query("probability_desc"),
    limit: int = Query(25, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """Filtered, paginated browse over the scored real population."""
    _require_population()
    return real_service.browse(
        query=query, risk=risk, auto_renew=auto_renew, activity=activity,
        min_probability=min_probability, max_probability=max_probability,
        limit=limit, offset=offset, sort=sort)


# ------------------------------------------------ slash-safe real lookups
#
# KKBOX subscriber ids are base64 and roughly half of them contain "/".
# Starlette percent-decodes a path segment before routing, so even a correctly
# encoded %2F splits the path and the route never matches -- those subscribers
# 404 on the path-parameter form. These query-parameter variants take the id
# out of the path entirely. The path routes above are kept so existing callers
# keep working for ids that have no slash.

@app.post("/predict-real", response_model=RealPredictResponse)
def predict_real_query(msno: str = Query(..., description="Subscriber id")):
    """Predict by subscriber id passed as a query parameter."""
    msno = _validate_msno(msno)
    _require_real_ready()
    try:
        return real_service.predict(msno)
    except KeyError:
        raise HTTPException(status_code=404,
                            detail=f"Subscriber {msno} not found in the real dataset")
    except RuntimeError as e:
        log.error("real prediction failed for %s: %s", msno, e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/subscriber-real/analysis")
def subscriber_real_analysis_query(msno: str = Query(..., description="Subscriber id")):
    """Full real-mode analysis by subscriber id passed as a query parameter."""
    msno = _validate_msno(msno)
    _require_real_ready()
    try:
        return real_service.analysis(msno)
    except KeyError:
        raise HTTPException(status_code=404,
                            detail=f"Subscriber {msno} not found in the real dataset")
    except RuntimeError as e:
        log.error("real analysis failed for %s: %s", msno, e)
        raise HTTPException(status_code=500, detail=str(e))
