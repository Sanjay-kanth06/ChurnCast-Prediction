"""FastAPI contract: health, model info, feature lookup and both predict paths."""
from __future__ import annotations


from src import config
from src.features import FEATURE_COLUMNS

REQUIRED_PREDICTION_FIELDS = {"msno", "churn_probability", "predicted_label",
                              "risk_level", "model_version"}


# --------------------------------------------------------------- health

def test_health_returns_200(api_client):
    assert api_client.get("/health").status_code == 200


def test_health_reports_model_available(api_client):
    body = api_client.get("/health").json()
    assert body["status"] == "ok", f"service degraded: {body.get('detail')}"
    assert body["model_available"] is True
    assert body["feature_table_available"] is True
    assert body["model_version"] is not None


# --------------------------------------------------------------- model info

def test_model_info(api_client):
    body = api_client.get("/model-info").json()
    assert body["model_name"] == config.MLFLOW_MODEL_NAME
    assert body["feature_count"] == len(FEATURE_COLUMNS)
    assert body["mlflow_connected"] is True
    assert body["model_version"] is not None


def test_model_info_exposes_risk_thresholds(api_client):
    body = api_client.get("/model-info").json()
    rt = body["risk_thresholds"]
    assert rt["low_below"] == config.CHURN_RISK_LOW_THRESHOLD
    assert rt["high_at_or_above"] == config.CHURN_RISK_HIGH_THRESHOLD


# --------------------------------------------------------------- lookup

def test_sample_subscribers(api_client):
    body = api_client.get("/subscribers/sample").json()
    assert len(body["subscribers"]) >= 1


def test_subscriber_features(api_client, sample_msno):
    body = api_client.get(f"/subscriber/{sample_msno}/features").json()
    assert body["msno"] == sample_msno
    assert set(body["features"]) == set(FEATURE_COLUMNS)


def test_subscriber_features_404(api_client):
    r = api_client.get("/subscriber/SYN999999/features")
    assert r.status_code == 404
    assert "not found" in r.json()["detail"].lower()


# --------------------------------------------------------------- predict/{msno}

def test_predict_by_msno_returns_200(api_client, sample_msno):
    assert api_client.post(f"/predict/{sample_msno}").status_code == 200


def test_predict_by_msno_response_shape(api_client, sample_msno):
    body = api_client.post(f"/predict/{sample_msno}").json()
    assert REQUIRED_PREDICTION_FIELDS.issubset(body)
    assert body["msno"] == sample_msno
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["predicted_label"] in (0, 1)
    assert body["risk_level"] in ("LOW", "MEDIUM", "HIGH")


def test_predict_by_msno_label_matches_threshold(api_client, sample_msno):
    body = api_client.post(f"/predict/{sample_msno}").json()
    expected = int(body["churn_probability"] >= config.DECISION_THRESHOLD)
    assert body["predicted_label"] == expected


def test_predict_by_msno_risk_band_matches_probability(api_client, sample_msno):
    body = api_client.post(f"/predict/{sample_msno}").json()
    assert body["risk_level"] == config.risk_level(body["churn_probability"])


def test_predict_unknown_msno_returns_404(api_client):
    r = api_client.post("/predict/SYN999999")
    assert r.status_code == 404
    assert r.json()["detail"] == "Subscriber SYN999999 not found"


def test_predict_blank_msno_is_rejected(api_client):
    assert api_client.post("/predict/%20").status_code == 422


def test_predictions_are_deterministic(api_client, sample_msno):
    a = api_client.post(f"/predict/{sample_msno}").json()
    b = api_client.post(f"/predict/{sample_msno}").json()
    assert a["churn_probability"] == b["churn_probability"]


def test_different_subscribers_can_differ(api_client, feature_table):
    """Distinct feature profiles should be allowed to produce distinct scores.
    Not forced -- simply checks the model is not emitting a constant."""
    ids = [str(m) for m in feature_table.index[:12]]
    probs = {m: api_client.post(f"/predict/{m}").json()["churn_probability"] for m in ids}
    assert len(set(probs.values())) > 1, "every subscriber received an identical score"


# --------------------------------------------------------------- predict (features)

def test_predict_with_features(api_client):
    payload = {"msno": "SYN000001", "recency_days": 5, "txns_last_90d": 4,
               "active_days_last_30d": 18, "engagement_trend": 0.82,
               "plan_price": 149, "is_auto_renew": 1, "age": 24, "tenure_days": 540}
    r = api_client.post("/predict", json=payload)
    assert r.status_code == 200
    assert REQUIRED_PREDICTION_FIELDS.issubset(r.json())


def test_predict_rejects_negative_values(api_client):
    payload = {"msno": "SYN000001", "recency_days": -5}
    assert api_client.post("/predict", json=payload).status_code == 422


def test_predict_rejects_out_of_range_ratio(api_client):
    payload = {"msno": "SYN000001", "skip_ratio": 1.7}
    assert api_client.post("/predict", json=payload).status_code == 422


def test_predict_requires_msno(api_client):
    assert api_client.post("/predict", json={"recency_days": 5}).status_code == 422


def test_openapi_documents_all_endpoints(api_client):
    paths = api_client.get("/openapi.json").json()["paths"]
    for p in ["/health", "/model-info", "/predict", "/predict/{msno}"]:
        assert p in paths, f"{p} missing from OpenAPI schema"
