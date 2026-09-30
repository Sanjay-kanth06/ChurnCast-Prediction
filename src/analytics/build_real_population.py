"""
Batch scoring of the full real subscriber population.

The dashboard needs company-level numbers over 970,960 subscribers. Scoring
that on every page load is not viable, so this runs once offline and writes two
artifacts the API then serves as static reads.

DEFINITIONS ARE TIED TO EXISTING FEATURES, NOT INVENTED
--------------------------------------------------------
Every headline number resolves to a column that already exists in
features.parquet or to a policy already in src/config.py:

  active listener   log_active_days_last_30d > 0
                    -- had at least one listening session in the 30 days
                    before the observation cutoff.
  likely to churn   predicted_label == 1, i.e. probability >=
                    config.DECISION_THRESHOLD. The model's own call.
  high risk         config.risk_level(p) == "HIGH".
  auto-renew        current_auto_renew, which is 1/0 or missing. Missing stays
                    "unknown" rather than being folded into "disabled".
  activity band     thresholds on log_active_days_last_30d, stated below and
                    carried into the artifact so the UI never invents its own.

NO FABRICATED TRENDS
--------------------
There is one observation period, so population-level "this month vs last
month" cannot be computed and is not emitted. The one genuine trend available
is per-subscriber: log_secs_trend_ratio compares each subscriber's last 30 days
of listening against the 30 before it. That is aggregated into a distribution
and labelled as an engagement trend, which is what it is -- not a churn trend
and not a subscriber-count trend.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.api import real_service

log = logging.getLogger("build_real_population")

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data" / "processed" / "real_v2"
FEATURES_PARQUET = DATA_DIR / "features.parquet"
PREDICTIONS_PARQUET = DATA_DIR / "population_predictions.parquet"
ANALYTICS_JSON = DATA_DIR / "population_analytics.json"

TARGET = "is_churn"
ID = "msno"
SCORE_CHUNK = 100_000

# Activity bands over log_active_days_last_30d (max 30). Thresholds are stated
# here once and shipped in the artifact so the interface cannot drift from them.
ACTIVITY_BANDS = [
    ("High activity", 20, 31),
    ("Moderate activity", 8, 20),
    ("Low activity", 1, 8),
    ("Inactive", 0, 1),
]
ACTIVITY_FEATURE = "log_active_days_last_30d"
ACTIVE_LISTENER_RULE = f"{ACTIVITY_FEATURE} > 0"

PROB_BUCKETS = [(i / 10, (i + 1) / 10) for i in range(10)]


def _counts(series: pd.Series, order: list[str]) -> dict[str, int]:
    vc = series.value_counts()
    return {k: int(vc.get(k, 0)) for k in order}


def _pct(part: int, whole: int) -> float:
    return round(part / whole, 6) if whole else 0.0


def score_population(model, features: pd.DataFrame,
                     columns: list[str]) -> tuple[pd.DataFrame, float]:
    """Probability for every subscriber, in chunks.

    Chunked because the design constraint is a single offline pass over ~1M
    rows, not because the model needs it -- it keeps peak memory flat and makes
    progress visible on a job that takes minutes.
    """
    t0 = time.time()
    parts = []
    total = len(features)
    for start in range(0, total, SCORE_CHUNK):
        block = features.iloc[start:start + SCORE_CHUNK]
        aligned = block.loc[:, columns]
        parts.append(model.predict_proba(aligned)[:, 1])
        if (start // SCORE_CHUNK) % 3 == 0:
            log.info("  scored %s / %s", f"{min(start + SCORE_CHUNK, total):,}",
                     f"{total:,}")
    probability = np.concatenate(parts)
    return probability, time.time() - t0


def top_signals(model, features: pd.DataFrame, columns: list[str],
                chunk: int = 50_000) -> tuple[np.ndarray, np.ndarray]:
    """The single strongest risk-raising feature per subscriber.

    Uses the model's own pred_contribs (exact SHAP in log-odds space) and keeps
    only the argmax, so the at-risk table can show a real "top risk factor"
    without persisting a 970,960 x 58 contribution matrix.
    """
    import xgboost as xgb

    pre = model.named_steps["preprocess"]
    clf = model.named_steps["clf"]
    booster = clf.get_booster()
    names = list(pre.get_feature_names_out())

    # Map encoded columns back onto their origin feature.
    origin = []
    for name in names:
        matched = name if name in columns else None
        if matched is None:
            for cat in ("city", "registered_via", "gender"):
                if name.startswith(f"{cat}_"):
                    matched = cat
                    break
        origin.append(matched)
    uniq = sorted({o for o in origin if o})
    index_of = {f: i for i, f in enumerate(uniq)}
    group = np.array([index_of.get(o, -1) for o in origin])

    best_feature = np.empty(len(features), dtype=object)
    best_value = np.zeros(len(features), dtype="float32")

    for start in range(0, len(features), chunk):
        block = features.iloc[start:start + chunk].loc[:, columns]
        matrix = pre.transform(block)
        matrix = matrix.toarray() if hasattr(matrix, "toarray") else matrix
        contribs = booster.predict(
            xgb.DMatrix(matrix, feature_names=names), pred_contribs=True)[:, :-1]

        collapsed = np.zeros((contribs.shape[0], len(uniq)), dtype="float32")
        for gi in range(len(uniq)):
            cols = np.where(group == gi)[0]
            if len(cols):
                collapsed[:, gi] = contribs[:, cols].sum(axis=1)

        idx = collapsed.argmax(axis=1)          # strongest RISK-RAISING signal
        rows = np.arange(collapsed.shape[0])
        best_feature[start:start + len(idx)] = [uniq[i] for i in idx]
        best_value[start:start + len(idx)] = collapsed[rows, idx]
        log.info("  signals %s / %s", f"{min(start + chunk, len(features)):,}",
                 f"{len(features):,}")

    return best_feature, best_value


def build(write: bool = True, with_signals: bool = True) -> dict:
    t_start = time.time()
    real_service.startup()
    if real_service.state["model"] is None:
        raise RuntimeError(real_service.state["load_error"] or "model unavailable")

    model = real_service.state["model"]
    columns = real_service.state["feature_columns"]
    features = real_service.state["features"]
    log.info("population: %s subscribers, %s features",
             f"{len(features):,}", len(columns))

    probability, score_seconds = score_population(model, features, columns)
    log.info("scored in %.1fs", score_seconds)

    frame = pd.DataFrame(index=features.index)
    frame["churn_probability"] = probability
    frame["predicted_label"] = (probability >= config.DECISION_THRESHOLD).astype("int8")
    frame["risk_level"] = pd.Series(
        [config.risk_level(p) for p in probability], index=features.index)
    frame["severity"] = pd.Series(
        [config.recommendation_severity(p) for p in probability],
        index=features.index)

    # Minimum extra fields the dashboard filters and analytics need.
    activity = pd.to_numeric(features.get(ACTIVITY_FEATURE), errors="coerce")
    frame["active_days_30d"] = activity
    frame["active_listener"] = (activity.fillna(0) > 0).astype("int8")
    renew = pd.to_numeric(features.get("current_auto_renew"), errors="coerce")
    frame["auto_renew"] = renew
    frame["gender"] = features["gender"].astype(str)
    frame["trend_ratio"] = pd.to_numeric(
        features.get("log_secs_trend_ratio"), errors="coerce")

    def band(v: float) -> str:
        if pd.isna(v):
            return "Inactive"
        for name, lo, hi in ACTIVITY_BANDS:
            if lo <= v < hi:
                return name
        return "High activity"

    frame["activity_band"] = activity.map(band)

    signal_seconds = 0.0
    if with_signals:
        t0 = time.time()
        best_feature, best_value = top_signals(model, features, columns)
        frame["top_signal"] = best_feature
        frame["top_signal_contribution"] = np.round(best_value, 4)
        signal_seconds = time.time() - t0
        log.info("top signals in %.1fs", signal_seconds)

    total = int(len(frame))
    risk_counts = _counts(frame["risk_level"], ["LOW", "MEDIUM", "HIGH"])
    severity_order = ["NONE", "EARLY_ENGAGEMENT", "TARGETED",
                      "HIGH_PRIORITY", "CRITICAL"]

    gender_order = ["female", "male", "unknown"]
    gender_counts = _counts(frame["gender"], gender_order)

    risk_by_gender = {}
    for g in gender_order:
        sub = frame[frame["gender"] == g]
        risk_by_gender[g] = {
            **_counts(sub["risk_level"], ["LOW", "MEDIUM", "HIGH"]),
            "total": int(len(sub)),
        }

    renew_groups = {
        "enabled": int((frame["auto_renew"] == 1).sum()),
        "disabled": int((frame["auto_renew"] == 0).sum()),
        "unknown": int(frame["auto_renew"].isna().sum()),
    }
    risk_by_auto_renew = {}
    for key, mask in (("enabled", frame["auto_renew"] == 1),
                      ("disabled", frame["auto_renew"] == 0),
                      ("unknown", frame["auto_renew"].isna())):
        sub = frame[mask]
        risk_by_auto_renew[key] = {
            **_counts(sub["risk_level"], ["LOW", "MEDIUM", "HIGH"]),
            "total": int(len(sub)),
            "mean_probability": round(float(sub["churn_probability"].mean()), 6)
            if len(sub) else None,
        }

    histogram = []
    for lo, hi in PROB_BUCKETS:
        n = int(((frame["churn_probability"] >= lo)
                 & (frame["churn_probability"] < (hi if hi < 1 else 1.0000001))).sum())
        histogram.append({"from": round(lo, 2), "to": round(hi, 2),
                          "count": n, "share": _pct(n, total)})

    activity_bands = []
    for name, _lo, _hi in ACTIVITY_BANDS:
        sub = frame[frame["activity_band"] == name]
        activity_bands.append({
            "band": name, "count": int(len(sub)), "share": _pct(len(sub), total),
            "mean_probability": round(float(sub["churn_probability"].mean()), 6)
            if len(sub) else None,
            **_counts(sub["risk_level"], ["LOW", "MEDIUM", "HIGH"]),
        })

    # Genuine engagement trend: per-subscriber 30d vs prior 30d listening.
    trend = frame["trend_ratio"].dropna()
    trend_buckets = [
        ("Sharp decline", -np.inf, 0.5), ("Decline", 0.5, 0.9),
        ("Stable", 0.9, 1.1), ("Growth", 1.1, 2.0), ("Strong growth", 2.0, np.inf),
    ]
    engagement_trend = [
        {"band": name, "count": int(((trend >= lo) & (trend < hi)).sum()),
         "share": _pct(int(((trend >= lo) & (trend < hi)).sum()), len(trend))}
        for name, lo, hi in trend_buckets
    ]

    details = real_service.model_details()
    top_features = details.get("top_features", [])[:10]

    active_listeners = int(frame["active_listener"].sum())
    likely = int(frame["predicted_label"].sum())
    high_risk = risk_counts["HIGH"]
    mean_prob = float(frame["churn_probability"].mean())

    analytics = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "provenance": {
            "data_mode": real_service.DATA_MODE,
            "observation_cutoff": real_service.OBSERVATION_CUTOFF,
            "model": real_service.MODEL_NAME,
            "model_version": real_service.state["model_version"],
            "population": total,
            "features_parquet": str(FEATURES_PARQUET),
            "decision_threshold": config.DECISION_THRESHOLD,
            "risk_thresholds": {
                "low_below": config.CHURN_RISK_LOW_THRESHOLD,
                "high_at_or_above": config.CHURN_RISK_HIGH_THRESHOLD,
            },
            "scoring_seconds": round(score_seconds, 1),
            "signal_seconds": round(signal_seconds, 1),
        },
        "definitions": {
            "active_listener": ACTIVE_LISTENER_RULE,
            "likely_to_churn": f"churn_probability >= {config.DECISION_THRESHOLD}",
            "high_risk": f"churn_probability >= {config.CHURN_RISK_HIGH_THRESHOLD}",
            "activity_bands": [{"band": b, "min_active_days_30d": lo,
                                "max_active_days_30d_exclusive": hi}
                               for b, lo, hi in ACTIVITY_BANDS],
            "trend_note": ("Single observation period: no company-level "
                           "period-over-period trend exists. The engagement "
                           "trend below is per-subscriber listening time in the "
                           "last 30 days versus the 30 before it."),
        },
        "kpi": {
            "total_subscribers": total,
            "active_listeners": active_listeners,
            "active_listeners_share": _pct(active_listeners, total),
            "likely_to_churn": likely,
            "likely_to_churn_share": _pct(likely, total),
            "high_risk_subscribers": high_risk,
            "high_risk_share": _pct(high_risk, total),
            "average_churn_probability": round(mean_prob, 6),
            "median_churn_probability": round(
                float(frame["churn_probability"].median()), 6),
            "auto_renew_enabled": renew_groups["enabled"],
            "auto_renew_enabled_share": _pct(renew_groups["enabled"], total),
        },
        "risk_distribution": [
            {"level": lvl, "count": risk_counts[lvl],
             "share": _pct(risk_counts[lvl], total)}
            for lvl in ("LOW", "MEDIUM", "HIGH")
        ],
        "severity_distribution": [
            {"severity": s, "count": int((frame["severity"] == s).sum()),
             "share": _pct(int((frame["severity"] == s).sum()), total)}
            for s in severity_order
        ],
        "auto_renew": renew_groups,
        "risk_by_auto_renew": risk_by_auto_renew,
        "gender": gender_counts,
        "risk_by_gender": risk_by_gender,
        "probability_histogram": histogram,
        "activity_segmentation": activity_bands,
        "engagement_trend": engagement_trend,
        "top_model_features": top_features,
        "elapsed_seconds": round(time.time() - t_start, 1),
    }

    if write:
        keep = ["churn_probability", "predicted_label", "risk_level", "severity",
                "active_days_30d", "active_listener", "auto_renew", "gender",
                "activity_band", "trend_ratio"]
        if with_signals:
            keep += ["top_signal", "top_signal_contribution"]
        out = frame[keep].reset_index()
        out.to_parquet(PREDICTIONS_PARQUET, index=False)
        ANALYTICS_JSON.write_text(json.dumps(analytics, indent=2), encoding="utf-8")
        log.info("wrote %s (%s rows) and %s", PREDICTIONS_PARQUET,
                 f"{len(out):,}", ANALYTICS_JSON)

    return analytics


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-signals", action="store_true",
                        help="skip per-subscriber top-signal attribution")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    analytics = build(write=not args.dry_run, with_signals=not args.no_signals)
    print(json.dumps({"provenance": analytics["provenance"],
                      "kpi": analytics["kpi"],
                      "risk_distribution": analytics["risk_distribution"]},
                     indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
