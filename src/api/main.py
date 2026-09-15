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
from fastapi import FastAPI, HTTPException, Path as PathParam
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src import config
from src.features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("api")

_state: dict[str, Any] = {
    "model": None, "model_version": None, "model_type": None,
    "features": None, "defaults": None, "load_error": None,
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
