"""Central configuration for the ChurnCast prototype.

Every tunable path, threshold and identifier lives here so that nothing is
duplicated across modules. Values may be overridden through environment
variables, which is how the Docker services configure themselves.
"""
from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------- paths
PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = Path(os.environ.get("CHURNCAST_RAW_DIR", PROJECT_ROOT / "data" / "raw"))
PROCESSED_DIR = Path(os.environ.get("CHURNCAST_PROCESSED_DIR", PROJECT_ROOT / "data" / "processed"))
REPORTS_DIR = Path(os.environ.get("CHURNCAST_REPORTS_DIR", PROJECT_ROOT / "reports"))

FEATURE_TABLE = PROCESSED_DIR / "features.parquet"
FEATURE_TABLE_CSV = PROCESSED_DIR / "features.csv"

RAW_FILES = {
    "train": RAW_DIR / "train.csv",
    "members": RAW_DIR / "members.csv",
    "transactions": RAW_DIR / "transactions.csv",
    "user_logs": RAW_DIR / "user_logs.csv",
}

# --------------------------------------------------------------- reproducibility
SEED = int(os.environ.get("CHURNCAST_SEED", 42))

# --------------------------------------------------------------- synthetic dataset
N_SUBSCRIBERS = int(os.environ.get("CHURNCAST_N_SUBSCRIBERS", 7500))
MSNO_PREFIX = "SYN"
MSNO_WIDTH = 6

# Each subscriber's membership expires in one of these months. The expiry date is
# that subscriber's observation cutoff, and churn is decided in the 30 days after
# it -- so cohorts give us a genuine temporal axis to split on.
COHORT_MONTHS = ["2024-07", "2024-08", "2024-09", "2024-10", "2024-11", "2024-12"]
TRAIN_COHORTS = COHORT_MONTHS[:4]   # 2024-07 .. 2024-10
VAL_COHORTS = COHORT_MONTHS[4:5]    # 2024-11
TEST_COHORTS = COHORT_MONTHS[5:]    # 2024-12

CHURN_WINDOW_DAYS = 30              # renewal must occur within this many days of expiry

# --------------------------------------------------------------- feature windows
SHORT_WINDOW_DAYS = 30
LONG_WINDOW_DAYS = 90
TREND_WINDOW_DAYS = 28

# --------------------------------------------------------------- MLflow
MLFLOW_TRACKING_URI = os.environ.get(
    "MLFLOW_TRACKING_URI", f"sqlite:///{(PROJECT_ROOT / 'mlflow.db').as_posix()}"
)
MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "ChurnCast")
MLFLOW_MODEL_NAME = os.environ.get("MLFLOW_MODEL_NAME", "ChurnCastModel")

# --------------------------------------------------------------- risk banding
# Configurable once, consumed by the API and the frontend via /model-info.
CHURN_RISK_LOW_THRESHOLD = float(os.environ.get("CHURN_RISK_LOW_THRESHOLD", 0.40))
CHURN_RISK_HIGH_THRESHOLD = float(os.environ.get("CHURN_RISK_HIGH_THRESHOLD", 0.70))
DECISION_THRESHOLD = float(os.environ.get("CHURNCAST_DECISION_THRESHOLD", 0.50))


def risk_level(probability: float) -> str:
    """Map a churn probability onto a human-readable band."""
    if probability < CHURN_RISK_LOW_THRESHOLD:
        return "LOW"
    if probability < CHURN_RISK_HIGH_THRESHOLD:
        return "MEDIUM"
    return "HIGH"


# ------------------------------------------------- recommendation policy
# A BUSINESS-RULE layer sitting on top of the model, not a model output.
#
# risk_level() above is the model's classification and is never altered by
# anything here. These bands only decide how strongly the interface suggests
# acting, so that a 0.72 and a 0.98 subscriber -- both correctly HIGH to the
# model -- do not receive identical operational urgency. Kept here rather than
# inline so the policy is tunable in one place.
RECOMMEND_TARGETED_THRESHOLD = float(
    os.environ.get("CHURNCAST_RECOMMEND_TARGETED", 0.70))
RECOMMEND_HIGH_PRIORITY_THRESHOLD = float(
    os.environ.get("CHURNCAST_RECOMMEND_HIGH_PRIORITY", 0.85))
RECOMMEND_CRITICAL_THRESHOLD = float(
    os.environ.get("CHURNCAST_RECOMMEND_CRITICAL", 0.95))

# Severities that authorise a retention intervention. LOW-risk subscribers are
# deliberately absent: a single declining feature must not trigger outreach
# when the model's overall estimate is low.
INTERVENTION_SEVERITIES = frozenset(
    {"EARLY_ENGAGEMENT", "TARGETED", "HIGH_PRIORITY", "CRITICAL"})


def recommendation_severity(probability: float) -> str:
    """How strongly to act, given the model's probability and risk band.

    Returns NONE, EARLY_ENGAGEMENT, TARGETED, HIGH_PRIORITY or CRITICAL. This
    is policy, not prediction: it never feeds back into risk_level().
    """
    band = risk_level(probability)
    if band == "LOW":
        return "NONE"
    if band == "MEDIUM":
        return "EARLY_ENGAGEMENT"
    # HIGH: graduate by probability inside the band.
    if probability >= RECOMMEND_CRITICAL_THRESHOLD:
        return "CRITICAL"
    if probability >= RECOMMEND_HIGH_PRIORITY_THRESHOLD:
        return "HIGH_PRIORITY"
    return "TARGETED"


# --------------------------------------------------------------- API
API_HOST = os.environ.get("CHURNCAST_API_HOST", "127.0.0.1")
API_PORT = int(os.environ.get("CHURNCAST_API_PORT", 8000))
FRONTEND_URL = os.environ.get("FRONTEND_URL", "http://localhost:3000")
CORS_ORIGINS = [o.strip() for o in os.environ.get(
    "CHURNCAST_CORS_ORIGINS", f"{FRONTEND_URL},http://127.0.0.1:3000"
).split(",") if o.strip()]


def msno(index: int) -> str:
    """Deterministic subscriber identifier, e.g. index 1 -> 'SYN000001'."""
    return f"{MSNO_PREFIX}{index:0{MSNO_WIDTH}d}"


def ensure_dirs() -> None:
    for d in (RAW_DIR, PROCESSED_DIR, REPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
