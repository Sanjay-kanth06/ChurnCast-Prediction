"""
Analytics support for the ChurnCast UI.

Everything here is derived from artifacts the system actually produced: the
persisted feature table, the registered model, and reports/model_metrics.json.
Nothing is synthesised for presentation. Where a value does not exist in the
data it is omitted rather than invented -- the feature table carries coded
identifiers (city, registration channel, payment method) and no names, no
registration dates and no usage history, so the UI can only show what is here.

The "why this prediction" signals are the model's own arithmetic, not a
heuristic: the registered model is a calibrated Logistic Regression, so each
feature's contribution to the log-odds is coefficient x standardised value,
averaged across the three calibration folds and summed back onto the original
feature names.
"""
from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd

from src import config
from src.api.recommendations import register_labels as _register_labels
from src.features import CATEGORICAL_FEATURES, FEATURE_COLUMNS

log = logging.getLogger("analytics")

# Features surfaced as human-readable signals, with the direction that raises risk.
SIGNAL_LABELS: dict[str, str] = {
    "recency_days": "Days since last transaction",
    "days_since_last_listen": "Days since last listening activity",
    "txns_last_30d": "Transactions in last 30 days",
    "txns_last_90d": "Transactions in last 90 days",
    "transaction_frequency": "Transaction frequency",
    "prior_cancellations": "Prior cancellations",
    "active_days_last_30d": "Active days in last 30 days",
    "active_days_last_90d": "Active days in last 90 days",
    "total_secs_last_30d": "Listening time, last 30 days",
    "total_secs_last_90d": "Listening time, last 90 days",
    "avg_daily_secs": "Average listening per active day",
    "unique_songs_last_30d": "Unique tracks, last 30 days",
    "unique_songs_last_90d": "Unique tracks, last 90 days",
    "skip_ratio": "Skip ratio",
    "completion_ratio": "Completion ratio",
    "engagement_trend": "Engagement trend",
    "recent_activity_ratio": "Recent vs baseline activity",
    "plan_price": "Plan price",
    "payment_amount": "Amount paid",
    "payment_consistency": "Payment consistency",
    "tenure_days": "Tenure",
    "age": "Age",
    "is_auto_renew": "Auto-renewal",
    "gender": "Gender",
    "city": "City",
    "registered_via": "Registration channel",
    "payment_method_last": "Payment method",
}

# How each feature is rendered. "secs" -> hours, "ratio" -> percent, etc.
_UNITS: dict[str, str] = {
    "recency_days": "days", "days_since_last_listen": "days", "tenure_days": "days",
    "total_secs_last_30d": "hours", "total_secs_last_90d": "hours",
    "avg_daily_secs": "hours", "skip_ratio": "pct", "completion_ratio": "pct",
    "recent_activity_ratio": "ratio", "payment_consistency": "pct",
}


def format_value(feature: str, value: Any) -> str:
    """Renders a raw feature value for display. Returns 'Not available' for
    missing data rather than substituting a placeholder number."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "Not available"
    unit = _UNITS.get(feature)
    if feature == "is_auto_renew":
        return {"1": "Enabled", "0": "Disabled"}.get(str(value), "Not available")
    if feature in ("city", "registered_via", "payment_method_last"):
        return "Not available" if str(value) == "unknown" else str(value)
    if feature == "gender":
        return "Not recorded" if str(value) == "unknown" else str(value).capitalize()
    if unit == "hours":
        return f"{float(value) / 3600:.1f} h"
    if unit == "pct":
        return f"{float(value) * 100:.1f}%"
    if unit == "days":
        return f"{int(round(float(value)))} days"
    if unit == "ratio":
        return f"{float(value):.2f}x"
    v = float(value)
    return f"{v:,.0f}" if abs(v) >= 100 else f"{v:,.2f}"


# ------------------------------------------------------------------ model metrics

def load_model_metrics() -> dict[str, Any] | None:
    """Reads reports/model_metrics.json, written by the training run."""
    path = config.REPORTS_DIR / "model_metrics.json"
    if not path.exists():
        log.warning("model_metrics.json not found at %s", path)
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:                                        # pragma: no cover
        log.warning("could not read model metrics: %s", e)
        return None


# ------------------------------------------------------------------ attribution

def build_attribution(model) -> dict[str, Any] | None:
    """Extracts the calibrated Logistic Regression's coefficients so that a
    per-feature log-odds contribution can be computed for any subscriber.

    Returns None for a model whose final estimator is not linear (e.g. if the
    gradient boosted candidate is ever promoted), in which case the UI omits the
    signals section rather than fabricating one.
    """
    try:
        subs = model.calibrated_classifiers_
        coefs, names = [], None
        for sub in subs:
            pipe = sub.estimator
            clf = pipe.named_steps["clf"]
            if not hasattr(clf, "coef_"):
                return None
            coefs.append(clf.coef_[0])
            if names is None:
                names = list(pipe.named_steps["preprocess"].get_feature_names_out())
        return {
            "coef": np.mean(coefs, axis=0),
            "names": names,
            "preprocess": subs[0].estimator.named_steps["preprocess"],
        }
    except Exception as e:
        log.warning("attribution unavailable: %s", e)
        return None


def _origin_feature(transformed_name: str) -> str | None:
    """Maps 'num__recency_days' / 'cat__city_13' back to the original feature."""
    if transformed_name.startswith("num__"):
        return transformed_name[5:]
    if transformed_name.startswith("cat__"):
        rest = transformed_name[5:]
        for cat in CATEGORICAL_FEATURES:
            if rest.startswith(cat + "_"):
                return cat
    return None


def explain(attribution: dict[str, Any] | None, row: pd.DataFrame,
            top_n: int = 6) -> list[dict[str, Any]]:
    """Per-feature log-odds contributions for one subscriber, strongest first.

    A positive contribution pushes the prediction toward churn, a negative one
    away from it. These are the model's own weights, but they describe the
    model's reasoning, not a causal mechanism in the world.
    """
    if attribution is None:
        return []
    try:
        X = attribution["preprocess"].transform(row[FEATURE_COLUMNS])
        X = np.asarray(X.todense()) if hasattr(X, "todense") else np.asarray(X)
        contrib = attribution["coef"] * X[0]

        per_feature: dict[str, float] = {}
        for name, c in zip(attribution["names"], contrib):
            origin = _origin_feature(name)
            if origin:
                per_feature[origin] = per_feature.get(origin, 0.0) + float(c)

        raw = row.iloc[0]
        out = []
        for feat, c in sorted(per_feature.items(), key=lambda kv: -abs(kv[1])):
            if abs(c) < 1e-9:
                continue
            out.append({
                "feature": feat,
                "label": SIGNAL_LABELS.get(feat, feat),
                "value": format_value(feat, raw.get(feat)),
                "contribution": round(c, 4),
                "direction": "increases" if c > 0 else "decreases",
            })
            if len(out) >= top_n:
                break
        return out
    except Exception as e:                                        # pragma: no cover
        log.warning("explain failed: %s", e)
        return []


# ------------------------------------------------------------------ recommendations
#
# Two separate ideas live here, and conflating them was the defect this
# replaces:
#
#   * model risk -- LOW / MEDIUM / HIGH, produced by config.risk_level() from
#     the calibrated probability. Never altered by anything below.
#   * severity   -- how hard to push operationally, from
#     config.recommendation_severity(). A business rule, not a model output.
#
# Two consequences. A 0.72 and a 0.98 subscriber are both HIGH to the model but
# earn TARGETED and CRITICAL severity respectively. And a 0.05 subscriber with
# one declining feature gets no intervention at all: the overall estimate takes
# precedence over any single signal.

_HEADLINES: dict[str, tuple[str, str]] = {
    "NONE": ("Subscriber appears stable",
             "No targeted retention intervention is indicated."),
    "EARLY_ENGAGEMENT": ("Suggested engagement steps",
                         "Early, low-intensity engagement is suggested."),
    "TARGETED": ("Recommended retention actions",
                 "Targeted retention outreach is suggested."),
    "HIGH_PRIORITY": ("Priority retention actions",
                      "This subscriber warrants prioritised attention."),
    "CRITICAL": ("Priority retention actions",
                 "This subscriber warrants the highest-priority attention."),
}


_register_labels(SIGNAL_LABELS)


def recommendation_policy(probability: float) -> dict[str, Any]:
    """The policy envelope for one prediction: severity, heading and wording.

    Kept separate from the recommendations themselves so the interface can
    describe the situation even when no feature rule fires.
    """
    band = config.risk_level(probability)
    severity = config.recommendation_severity(probability)
    headline, stance = _HEADLINES[severity]
    pct = f"{probability * 100:.1f}%"

    if severity == "NONE":
        summary = (f"Estimated churn probability is {pct}, placing this "
                   f"subscriber in the {band} band. No targeted retention "
                   f"intervention is indicated based on the current prediction. "
                   f"Maintain normal engagement and monitor for meaningful "
                   f"changes.")
    else:
        summary = (f"Estimated churn probability is {pct} ({band} risk). "
                   f"{stance} The steps below are drawn from this subscriber's "
                   f"own observed feature values.")

    return {
        "severity": severity,
        "risk_level": band,
        "headline": headline,
        "summary": summary,
        "intervention": severity in config.INTERVENTION_SEVERITIES,
        "thresholds": {
            "targeted_at": config.RECOMMEND_TARGETED_THRESHOLD,
            "high_priority_at": config.RECOMMEND_HIGH_PRIORITY_THRESHOLD,
            "critical_at": config.RECOMMEND_CRITICAL_THRESHOLD,
        },
    }


def _evidence(row: pd.Series, severity: str) -> list[dict[str, Any]]:
    """Feature rules that fire for this subscriber, before ordering.

    Every rule requires actual evidence in the feature values; none fires on
    the probability alone. `escalate` marks wording that only applies once
    severity has reached the higher bands.
    """
    out: list[dict[str, Any]] = []
    escalate = severity in ("HIGH_PRIORITY", "CRITICAL")

    def num(feature: str) -> float | None:
        v = row.get(feature)
        if v is None or (isinstance(v, float) and pd.isna(v)):
            return None
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    if str(row.get("is_auto_renew", "unknown")) == "0":
        out.append({
            "feature": "is_auto_renew", "weight": 4.0,
            "title": "Consider a renewal reminder",
            "detail": ("Auto-renewal is disabled, so the membership lapses "
                       "unless it is renewed manually. A reminder before the "
                       "renewal date is the most direct step available."),
            "basis": "is_auto_renew = Disabled",
        })

    cancels = num("prior_cancellations")
    if cancels is not None and cancels > 0:
        n = int(cancels)
        out.append({
            "feature": "prior_cancellations", "weight": 3.5,
            "title": ("Route to a retention specialist" if escalate
                      else "Account for cancellation history"),
            "detail": (f"There are {n} prior cancellation(s) on record. "
                       + ("Given the elevated estimate, a human retention "
                          "contact is preferable to an automated flow."
                          if escalate else
                          "An automated flow may be less effective here than "
                          "a personal contact.")),
            "basis": f"prior_cancellations = {n}",
        })

    trend = num("engagement_trend")
    if trend is not None and trend < 0:
        out.append({
            "feature": "engagement_trend", "weight": 3.0,
            "title": "Consider a re-engagement prompt",
            "detail": ("Listening activity is trending downward over the four "
                       "weeks before the observation cutoff. Personalised "
                       "content suggestions may help recover engagement."),
            "basis": f"engagement_trend = {trend:.1f}",
        })

    ratio = num("recent_activity_ratio")
    if ratio is not None and ratio < 0.75:
        out.append({
            "feature": "recent_activity_ratio", "weight": 2.5,
            "title": "Consider personalised engagement",
            "detail": ("Recent 30-day listening sits materially below this "
                       "subscriber's own 90-day average, which may indicate "
                       "reduced engagement."),
            "basis": f"recent_activity_ratio = {ratio:.2f}x",
        })

    idle = num("days_since_last_listen")
    if idle is not None and idle >= 14:
        out.append({
            "feature": "days_since_last_listen", "weight": 2.0,
            "title": "Review recent listening inactivity",
            "detail": (f"No listening activity was recorded in the {int(idle)} "
                       f"days before the observation cutoff."),
            "basis": f"days_since_last_listen = {int(idle)}",
        })

    return out


def recommend(probability: float, row: pd.Series,
              signals: list[dict[str, Any]] | None = None) -> list[dict[str, str]]:
    """Deterministic, rule-based suggestions for one subscriber.

    Gated on recommendation severity: a LOW-risk subscriber receives no
    intervention, however unflattering an individual feature looks. Where
    several rules fire they are ordered by the model's own attribution for the
    underlying feature when it is available, so the strongest real evidence
    leads; `weight` is only a fallback ordering.

    Wording stays provisional throughout. Nothing here claims an action will
    prevent churn.
    """
    policy = recommendation_policy(probability)
    severity = policy["severity"]

    if not policy["intervention"]:
        return [{
            "title": "Continue normal engagement",
            "detail": policy["summary"],
            "basis": (f"churn_probability = {probability:.3f} "
                      f"(< {config.CHURN_RISK_LOW_THRESHOLD})"),
        }]

    rules = _evidence(row, severity)

    strength = {s["feature"]: abs(float(s.get("contribution", 0.0)))
                for s in (signals or [])}
    rules.sort(key=lambda r: (-strength.get(r["feature"], 0.0), -r["weight"]))

    recs = [{"title": r["title"], "detail": r["detail"], "basis": r["basis"]}
            for r in rules]

    if not recs:
        recs.append({
            "title": "Monitor this subscriber",
            "detail": ("The estimate is elevated, but no individual feature "
                       "rule matched this profile. Review the model signals "
                       "before choosing an action."),
            "basis": f"churn_probability = {probability:.3f}",
        })

    return recs


def interpretation(probability: float) -> str:
    band = config.risk_level(probability)
    pct = f"{probability * 100:.1f}%"
    if band == "HIGH":
        return (f"Based on the current subscriber feature profile, the model estimates a "
                f"relatively high probability of churn ({pct}) within the prediction "
                f"horizon.")
    if band == "MEDIUM":
        return (f"Based on the current subscriber feature profile, the model estimates a "
                f"moderate probability of churn ({pct}) within the prediction horizon.")
    return (f"Based on the current subscriber feature profile, the model estimates a "
            f"relatively low probability of churn ({pct}) within the prediction horizon.")


# ------------------------------------------------------------------ batch scoring

def score_all(model, features: pd.DataFrame) -> pd.DataFrame:
    """Scores every subscriber once so the dashboard and subscriber list can be
    served without re-running the model per request. Computed at start-up and
    cached; the model does not change while the process is alive."""
    probs = model.predict_proba(features[FEATURE_COLUMNS])[:, 1]
    out = pd.DataFrame(index=features.index)
    out["churn_probability"] = np.round(probs, 4)
    out["predicted_label"] = (probs >= config.DECISION_THRESHOLD).astype(int)
    out["risk_level"] = [config.risk_level(p) for p in probs]
    log.info("scored %d subscribers for dashboard and browse views", len(out))
    return out


def risk_distribution(scored: pd.DataFrame) -> list[dict[str, Any]]:
    counts = scored["risk_level"].value_counts()
    total = int(len(scored))
    return [
        {"level": lvl,
         "count": int(counts.get(lvl, 0)),
         "share": round(float(counts.get(lvl, 0)) / total, 4) if total else 0.0}
        for lvl in ("LOW", "MEDIUM", "HIGH")
    ]


def profile(row: pd.Series) -> list[dict[str, str]]:
    """Subscriber profile from fields that genuinely exist in the feature table.

    The dataset carries coded identifiers rather than names: city and
    registration channel are integer codes, and there is no personal name, email
    or registration date to show. Codes are labelled as codes.
    """
    fields = [
        ("City", "city", "code"),
        ("Registration channel", "registered_via", "code"),
        ("Gender", "gender", None),
        ("Auto-renewal", "is_auto_renew", None),
        ("Payment method", "payment_method_last", "code"),
        ("Tenure", "tenure_days", None),
        ("Age", "age", None),
        ("Plan price", "plan_price", None),
    ]
    out = []
    for label, feat, kind in fields:
        raw = row.get(feat)
        shown = format_value(feat, raw)
        if kind == "code" and shown != "Not available":
            shown = f"Code {shown}"
        out.append({"label": label, "value": shown})
    return out
