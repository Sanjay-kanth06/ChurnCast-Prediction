"""
Tests for the batch population artifact and the endpoints that serve it.

The point of these is arithmetic integrity. Dashboard numbers are the kind that
get quoted in a meeting, so every breakdown must reconcile to the population
total and every displayed figure must be traceable to the artifact rather than
computed a second time in the browser. A segmentation that silently loses rows
would still render a convincing dashboard.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src import config

ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "data" / "processed" / "real_v2"
ANALYTICS_JSON = ARTIFACT_DIR / "population_analytics.json"
PREDICTIONS_PARQUET = ARTIFACT_DIR / "population_predictions.parquet"


@pytest.fixture(scope="module")
def analytics():
    if not ANALYTICS_JSON.exists():
        pytest.skip("population analytics not built")
    return json.loads(ANALYTICS_JSON.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def predictions():
    if not PREDICTIONS_PARQUET.exists():
        pytest.skip("population predictions not built")
    import pandas as pd
    return pd.read_parquet(PREDICTIONS_PARQUET)


def _skip_unless_population(api_client):
    res = api_client.get("/population/analytics")
    if res.status_code != 200:
        pytest.skip("population endpoints unavailable")
    return res.json()


# ------------------------------------------------------------- artifact shape

def test_population_row_count_matches_provenance(predictions, analytics):
    assert len(predictions) == analytics["provenance"]["population"]
    assert len(predictions) == analytics["kpi"]["total_subscribers"]


def test_every_subscriber_has_a_prediction(predictions):
    assert predictions["churn_probability"].notna().all()
    assert predictions["predicted_label"].notna().all()
    assert predictions["risk_level"].notna().all()
    assert predictions["msno"].nunique() == len(predictions)


def test_probabilities_are_in_range(predictions):
    p = predictions["churn_probability"]
    assert p.min() >= 0.0
    assert p.max() <= 1.0


def test_predicted_label_follows_the_decision_threshold(predictions):
    expected = (predictions["churn_probability"] >= config.DECISION_THRESHOLD)
    assert (predictions["predicted_label"].astype(bool) == expected).all()


def test_risk_level_follows_the_configured_policy(predictions):
    sample = predictions.sample(min(2000, len(predictions)), random_state=0)
    expected = sample["churn_probability"].map(config.risk_level)
    assert (sample["risk_level"] == expected).all()


# ---------------------------------------------------------------- totals

def test_risk_distribution_sums_to_the_population(analytics):
    total = analytics["kpi"]["total_subscribers"]
    assert sum(r["count"] for r in analytics["risk_distribution"]) == total


def test_severity_distribution_sums_to_the_population(analytics):
    total = analytics["kpi"]["total_subscribers"]
    assert sum(s["count"] for s in analytics["severity_distribution"]) == total


def test_gender_counts_sum_to_the_population(analytics):
    total = analytics["kpi"]["total_subscribers"]
    assert sum(analytics["gender"].values()) == total


def test_auto_renew_counts_sum_to_the_population(analytics):
    total = analytics["kpi"]["total_subscribers"]
    assert sum(analytics["auto_renew"].values()) == total


def test_risk_by_gender_reconciles_with_gender_totals(analytics):
    for gender, counts in analytics["risk_by_gender"].items():
        assert counts["LOW"] + counts["MEDIUM"] + counts["HIGH"] == counts["total"]
        assert counts["total"] == analytics["gender"][gender]


def test_risk_by_auto_renew_reconciles(analytics):
    for key, counts in analytics["risk_by_auto_renew"].items():
        assert counts["LOW"] + counts["MEDIUM"] + counts["HIGH"] == counts["total"]
        assert counts["total"] == analytics["auto_renew"][key]


def test_probability_histogram_sums_to_the_population(analytics):
    total = analytics["kpi"]["total_subscribers"]
    assert sum(b["count"] for b in analytics["probability_histogram"]) == total


def test_activity_segmentation_sums_to_the_population(analytics):
    total = analytics["kpi"]["total_subscribers"]
    assert sum(s["count"] for s in analytics["activity_segmentation"]) == total


def test_shares_are_consistent_with_their_counts(analytics):
    total = analytics["kpi"]["total_subscribers"]
    for r in analytics["risk_distribution"]:
        assert r["share"] == pytest.approx(r["count"] / total, abs=1e-6)


# ------------------------------------------------------- definitions are real

def test_headline_definitions_reference_actual_features(analytics):
    """Every KPI must resolve to a feature or policy, not a business rule
    invented for the dashboard."""
    d = analytics["definitions"]
    assert d["active_listener"] == "log_active_days_last_30d > 0"
    assert str(config.DECISION_THRESHOLD) in d["likely_to_churn"]
    assert str(config.CHURN_RISK_HIGH_THRESHOLD) in d["high_risk"]
    assert d["activity_bands"]


def test_active_listener_count_matches_its_definition(predictions, analytics):
    computed = int((predictions["active_days_30d"].fillna(0) > 0).sum())
    assert computed == analytics["kpi"]["active_listeners"]


def test_likely_to_churn_matches_the_model_threshold(predictions, analytics):
    computed = int(predictions["predicted_label"].sum())
    assert computed == analytics["kpi"]["likely_to_churn"]


def test_high_risk_matches_the_risk_policy(predictions, analytics):
    computed = int((predictions["risk_level"] == "HIGH").sum())
    assert computed == analytics["kpi"]["high_risk_subscribers"]


def test_average_probability_matches_the_predictions(predictions, analytics):
    assert analytics["kpi"]["average_churn_probability"] == pytest.approx(
        float(predictions["churn_probability"].mean()), abs=1e-5)


def test_no_period_over_period_company_trend_is_emitted(analytics):
    """A single observation period cannot support one, so none must appear."""
    assert "trend_note" in analytics["definitions"]
    blob = json.dumps(analytics).lower()
    for forbidden in ("vs_last_month", "vs_prior_period", "month_over_month",
                      "change_pct", "trend_pct"):
        assert forbidden not in blob


def test_provenance_records_the_real_model(analytics):
    p = analytics["provenance"]
    assert p["data_mode"] == "real_v2"
    assert p["observation_cutoff"] == "2017-02-28"
    assert p["model"] == "ChurnCastRealV2"
    assert p["model_version"] == "1"
    assert p["population"] == 970960


# ------------------------------------------------------------- endpoints

def test_analytics_endpoint_matches_the_artifact(api_client, analytics):
    served = _skip_unless_population(api_client)
    assert served["kpi"] == analytics["kpi"]
    assert served["risk_distribution"] == analytics["risk_distribution"]
    assert served["provenance"]["population"] == analytics["provenance"]["population"]


def test_at_risk_returns_the_highest_probabilities(api_client):
    _skip_unless_population(api_client)
    rows = api_client.get("/population/at-risk?limit=10").json()["rows"]
    assert len(rows) == 10
    probs = [r["churn_probability"] for r in rows]
    assert probs == sorted(probs, reverse=True)
    assert all(r["risk_level"] == "HIGH" for r in rows)
    assert all(0.0 <= r["churn_probability"] <= 1.0 for r in rows)


def test_at_risk_rows_carry_a_real_top_signal(api_client):
    _skip_unless_population(api_client)
    rows = api_client.get("/population/at-risk?limit=5").json()["rows"]
    for r in rows:
        assert r["top_signal"], "top signal must name an actual feature"
        assert r["top_signal_contribution"] is not None


def test_browse_total_matches_the_risk_distribution(api_client, analytics):
    _skip_unless_population(api_client)
    high = api_client.get("/population/subscribers?risk=HIGH&limit=1").json()
    expected = next(r["count"] for r in analytics["risk_distribution"]
                    if r["level"] == "HIGH")
    assert high["total"] == expected


def test_browse_unfiltered_total_is_the_population(api_client, analytics):
    _skip_unless_population(api_client)
    body = api_client.get("/population/subscribers?limit=1").json()
    assert body["total"] == analytics["kpi"]["total_subscribers"]


def test_browse_probability_filter_is_respected(api_client):
    _skip_unless_population(api_client)
    body = api_client.get(
        "/population/subscribers?min_probability=0.9&limit=20").json()
    assert all(r["churn_probability"] >= 0.9 for r in body["rows"])


def test_population_rows_declare_the_real_data_mode(api_client):
    _skip_unless_population(api_client)
    assert api_client.get("/population/subscribers?limit=1").json()[
        "data_mode"] == "real_v2"


# ------------------------------------------------------------ mode isolation

def test_population_endpoints_never_return_synthetic_ids(api_client):
    """Synthetic IDs are SYN-prefixed; none may appear in the real population."""
    _skip_unless_population(api_client)
    rows = api_client.get("/population/subscribers?limit=50").json()["rows"]
    assert not any(r["msno"].startswith("SYN") for r in rows)


def test_synthetic_dashboard_is_unaffected_by_the_population_artifact(api_client):
    body = api_client.get("/dashboard/summary").json()
    assert body["total_subscribers"] == 7500
    assert sum(r["count"] for r in body["risk_distribution"]) == 7500


def test_synthetic_and_real_populations_have_different_totals(api_client, analytics):
    _skip_unless_population(api_client)
    synthetic = api_client.get("/dashboard/summary").json()["total_subscribers"]
    assert synthetic == 7500
    assert analytics["kpi"]["total_subscribers"] == 970960
    assert synthetic != analytics["kpi"]["total_subscribers"]
