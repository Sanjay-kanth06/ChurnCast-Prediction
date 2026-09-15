"""
Data drift monitoring for the ChurnCast prototype, built on Evidently.

Written against the Evidently 0.7.x API (Report + presets + Dataset), which
differs substantially from the 0.4-era `Report(metrics=[...]).run(...)` /
`save_html` interface found in older tutorials. The installed version is
verified at import time so a mismatch fails loudly rather than silently
producing an empty report.

The reference dataset is the training cohort's feature table; the current
dataset is a later cohort. A controlled shift can be injected into the current
dataset to demonstrate that the monitoring actually detects drift -- that mode
is clearly labelled as a synthetic demonstration.
"""
from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from src import config
from src.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES

log = logging.getLogger("monitoring")

# Features highlighted in the drift report. Kept small so the report is readable.
MONITORED_FEATURES = [
    "recency_days",
    "txns_last_90d",
    "active_days_last_30d",
    "total_secs_last_30d",
    "engagement_trend",
    "plan_price",
    "tenure_days",
    "is_auto_renew",
]


def _evidently():
    """Imports the Evidently 0.7 API, reporting the installed version."""
    import evidently
    from evidently import DataDefinition, Dataset, Report
    from evidently.presets import DataDriftPreset
    log.info("evidently version %s", evidently.__version__)
    return evidently, Report, Dataset, DataDefinition, DataDriftPreset


def split_reference_current(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reference = training cohorts, current = the most recent cohort."""
    ref = table[table["cohort_month"].isin(config.TRAIN_COHORTS)]
    cur = table[table["cohort_month"].isin(config.TEST_COHORTS)]
    return ref, cur


def inject_synthetic_drift(df: pd.DataFrame, seed: int = config.SEED) -> pd.DataFrame:
    """SYNTHETIC DRIFT DEMONSTRATION ONLY.

    Applies a controlled, deliberate shift so the monitoring pipeline has
    something to detect. This does not represent observed production drift.
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    n = len(out)
    if "recency_days" in out:
        out["recency_days"] = out["recency_days"].fillna(0) * 1.9 + rng.normal(14, 4, n)
    if "active_days_last_30d" in out:
        out["active_days_last_30d"] = (out["active_days_last_30d"] * 0.55).clip(0, 31)
    if "total_secs_last_30d" in out:
        out["total_secs_last_30d"] = out["total_secs_last_30d"] * 0.6
    if "plan_price" in out:
        out["plan_price"] = out["plan_price"] * 1.35
    if "engagement_trend" in out:
        out["engagement_trend"] = out["engagement_trend"] - rng.normal(90, 30, n)
    if "is_auto_renew" in out:
        flip = rng.random(n) < 0.30
        out.loc[flip, "is_auto_renew"] = "0"
    return out


def generate_drift_report(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    output_path: Path | None = None,
    features: list[str] | None = None,
) -> dict:
    """Builds an Evidently data-drift report and writes it as HTML.

    Returns a summary dict: output path, feature count, and the number of
    features flagged as drifted.
    """
    _, Report, Dataset, DataDefinition, DataDriftPreset = _evidently()

    cols = features or MONITORED_FEATURES
    cols = [c for c in cols if c in reference.columns and c in current.columns]
    if not cols:
        raise ValueError("no monitored features present in both frames")

    num = [c for c in cols if c in NUMERIC_FEATURES]
    cat = [c for c in cols if c in CATEGORICAL_FEATURES]

    # reset_index is required: the feature table is indexed by msno (a string),
    # and Evidently otherwise tries to treat that index as a time axis and fails
    # inside its plotting layer with a dtype promotion error.
    ref = reference[cols].copy().reset_index(drop=True)
    cur = current[cols].copy().reset_index(drop=True)
    for c in cat:
        ref[c] = ref[c].astype(str)
        cur[c] = cur[c].astype(str)

    definition = DataDefinition(numerical_columns=num, categorical_columns=cat)
    ref_ds = Dataset.from_pandas(ref, data_definition=definition)
    cur_ds = Dataset.from_pandas(cur, data_definition=definition)

    report = Report(metrics=[DataDriftPreset()])
    run = report.run(current_data=cur_ds, reference_data=ref_ds)

    output_path = output_path or (config.REPORTS_DIR /
                                  f"drift_report_{date.today().isoformat()}.html")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run.save_html(str(output_path))

    summary = _summarise(run, cols)
    summary["output_path"] = str(output_path)
    log.info("drift report written to %s (%d/%d features drifted)",
             output_path, summary["drifted_features"], summary["n_features"])
    return summary


def _summarise(run, cols: list[str]) -> dict:
    """Pulls the drifted-column count out of the run payload.

    In Evidently 0.7 the DataDriftPreset emits a DriftedColumnsCount metric
    whose value is a {'count': n, 'share': s} mapping, followed by one
    per-column drift score. The count entry is located by shape rather than by
    a hard-coded index, so a change in metric ordering does not break this.
    """
    drifted, share, per_column = None, None, {}
    try:
        metrics = run.dict().get("metrics", [])
        scores = []
        for m in metrics:
            v = m.get("value")
            if isinstance(v, dict) and "count" in v and "share" in v:
                drifted, share = int(v["count"]), float(v["share"])
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                scores.append(float(v))
        # the per-column scores follow the count entry, in monitored-feature order
        per_column = {c: round(s, 6) for c, s in zip(cols, scores)}
    except Exception as e:                                    # pragma: no cover
        log.warning("could not summarise drift payload: %s", e)

    return {"n_features": len(cols), "features": cols,
            "drifted_features": drifted if drifted is not None else -1,
            "share_drifted": share,
            "per_column_drift_score": per_column}
