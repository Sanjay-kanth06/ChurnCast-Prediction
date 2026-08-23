"""
W2-T5: FastAPI service. Loads the churncast-model from the MLflow Model
Registry at the stage named by MODEL_STAGE (default "Staging") at startup,
and exposes /predict and /health per the contract in §6.

Run:
    MLFLOW_TRACKING_URI=sqlite:///mlflow.db MODEL_STAGE=Staging \\
        uvicorn src.api.main:app --port 8000
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from typing import Optional

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("api")

REGISTERED_MODEL_NAME = "churncast-model"
MODEL_STAGE = os.environ.get("MODEL_STAGE", "Staging")

_state: dict = {"model": None, "model_version": None}


def _load_model() -> None:
    client = mlflow.tracking.MlflowClient()
    versions = client.get_latest_versions(REGISTERED_MODEL_NAME, stages=[MODEL_STAGE])
    if not versions:
        raise RuntimeError(
            f"No model version found for '{REGISTERED_MODEL_NAME}' at stage '{MODEL_STAGE}'. "
            "Run `python -m src.models.train` first."
        )
    mv = versions[0]
    model_uri = f"models:/{REGISTERED_MODEL_NAME}/{mv.version}"
    _state["model"] = mlflow.sklearn.load_model(model_uri)
    _state["model_version"] = mv.version
    log.info("loaded %s v%s (stage=%s)", REGISTERED_MODEL_NAME, mv.version, MODEL_STAGE)


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        _load_model()
    except Exception as e:  # keep the process up so /health can report the problem
        log.error("model load failed at startup: %s", e)
    yield


app = FastAPI(title="ChurnCast API", version="1.0", lifespan=lifespan)


class PredictRequest(BaseModel):
    msno: str
    recency_days: float
    txns_last_90d: int
    active_days_last_30d: int
    engagement_trend: float
    plan_price: float
    is_auto_renew: bool
    age: Optional[float] = None
    tenure_days: float
    # feature_spec.yaml also includes days_since_last_listen, gender,
    # prior_cancellations, registered_via, which the §6 sample payload omits;
    # accept them optionally so the contract stays a strict superset-safe fit.
    days_since_last_listen: Optional[float] = Field(default=0)
    gender: Optional[str] = Field(default="unknown")
    prior_cancellations: Optional[int] = Field(default=0)
    registered_via: Optional[str] = Field(default="unknown")


class PredictResponse(BaseModel):
    msno: str
    churn_probability: float
    model_version: str
    predicted_label: int


@app.get("/health")
def health():
    return {
        "status": "ok" if _state["model"] is not None else "model_unavailable",
        "model_stage": MODEL_STAGE,
        "model_version": _state["model_version"],
    }


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check /health.")

    row = pd.DataFrame(
        [
            {
                "recency_days": req.recency_days,
                "days_since_last_listen": req.days_since_last_listen,
                "txns_last_90d": req.txns_last_90d,
                "active_days_last_30d": req.active_days_last_30d,
                "engagement_trend": req.engagement_trend,
                "plan_price": req.plan_price,
                "prior_cancellations": req.prior_cancellations,
                "age": req.age,
                "tenure_days": req.tenure_days,
                "gender": req.gender,
                "registered_via": req.registered_via,
                "is_auto_renew": str(int(req.is_auto_renew)),
            }
        ]
    )

    prob = float(_state["model"].predict_proba(row)[0, 1])
    return PredictResponse(
        msno=req.msno,
        churn_probability=round(prob, 4),
        model_version=str(_state["model_version"]),
        predicted_label=int(prob >= 0.5),
    )
