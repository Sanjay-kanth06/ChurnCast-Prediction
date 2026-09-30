"""
Regression tests for the real (v2) prediction route.

Two things these are really guarding. First, route isolation: the synthetic
/predict path and the real /predict-real path must not reach into each other's
model or feature table, because they score different feature spaces and a
crossed wire would return a confident, wrong, entirely plausible number.
Second, train/serve column alignment, which this project has already shipped
wrong twice and which never announces itself at runtime.
"""
from __future__ import annotations

import pytest

from src import config
from src.api import real_service

REQUIRED_FIELDS = {"msno", "churn_probability", "predicted_label", "risk_level",
                   "model_version", "model_name", "data_mode"}


def _skip_if_unavailable(api_client):
    body = api_client.get("/health/real").json()
    if not (body["model_available"] and body["feature_table_available"]):
        pytest.skip(f"real model not available: {body.get('detail')}")
    return body


@pytest.fixture(scope="module")
def real_msno(api_client):
    _skip_if_unavailable(api_client)
    body = api_client.get("/subscribers-real/sample?n=3").json()
    return body["subscribers"][0]


# ------------------------------------------------------------------ health

def test_real_health_reports_the_registered_model(api_client):
    body = _skip_if_unavailable(api_client)
    assert body["model_name"] == "ChurnCastRealV2"
    assert body["model_version"] is not None
    assert body["data_mode"] == "real_v2"
    assert body["observation_cutoff"] == "2017-02-28"
    assert body["subscribers"] == 970960
    assert body["feature_count"] == 57


# -------------------------------------------------------- known subscriber

def test_known_real_subscriber_scores(api_client, real_msno):
    res = api_client.post(f"/predict-real/{real_msno}")
    assert res.status_code == 200
    body = res.json()
    assert REQUIRED_FIELDS <= set(body)
    assert body["msno"] == real_msno


def test_probability_is_a_float_in_range(api_client, real_msno):
    body = api_client.post(f"/predict-real/{real_msno}").json()
    p = body["churn_probability"]
    assert isinstance(p, float)
    assert 0.0 <= p <= 1.0


def test_label_and_risk_band_agree_with_the_probability(api_client, real_msno):
    body = api_client.post(f"/predict-real/{real_msno}").json()
    p = body["churn_probability"]
    assert body["predicted_label"] == int(p >= config.DECISION_THRESHOLD)
    assert body["risk_level"] == config.risk_level(p)
    assert body["risk_level"] in {"LOW", "MEDIUM", "HIGH"}


def test_response_names_the_real_model_and_mode(api_client, real_msno):
    body = api_client.post(f"/predict-real/{real_msno}").json()
    assert body["model_name"] == "ChurnCastRealV2"
    assert body["data_mode"] == "real_v2"
    assert body["model_version"] is not None


def test_predictions_are_deterministic(api_client, real_msno):
    a = api_client.post(f"/predict-real/{real_msno}").json()
    b = api_client.post(f"/predict-real/{real_msno}").json()
    assert a["churn_probability"] == b["churn_probability"]


# ------------------------------------------------------------ bad input

def test_unknown_msno_returns_404(api_client):
    _skip_if_unavailable(api_client)
    res = api_client.post("/predict-real/NOT_A_REAL_SUBSCRIBER=")
    assert res.status_code == 404
    assert "not found" in res.json()["detail"].lower()


def test_unknown_msno_returns_no_probability(api_client):
    """A miss must not be answered with a number."""
    _skip_if_unavailable(api_client)
    body = api_client.post("/predict-real/NOT_A_REAL_SUBSCRIBER=").json()
    assert "churn_probability" not in body


def test_whitespace_msno_is_rejected(api_client):
    _skip_if_unavailable(api_client)
    res = api_client.post("/predict-real/%20")
    assert res.status_code == 422


def test_overlong_msno_is_rejected(api_client):
    _skip_if_unavailable(api_client)
    res = api_client.post("/predict-real/" + "x" * 200)
    assert res.status_code == 422


# ------------------------------------------------- feature-column alignment

def test_served_columns_match_the_trained_contract(api_client, real_msno):
    """The contract written at registration is what the model is fed."""
    _skip_if_unavailable(api_client)
    columns = real_service.state["feature_columns"]
    row = real_service.state["features"].loc[[real_msno]].drop(
        columns=["is_churn"], errors="ignore")
    aligned = real_service.align(row)
    assert list(aligned.columns) == columns
    assert len(columns) == 57


def test_alignment_rejects_a_missing_trained_column(real_msno):
    """A dropped column must raise, not be silently imputed."""
    row = real_service.state["features"].loc[[real_msno]].drop(
        columns=["is_churn"], errors="ignore")
    broken = row.drop(columns=[real_service.state["feature_columns"][0]])
    with pytest.raises(RuntimeError, match="missing"):
        real_service.align(broken)


def test_alignment_restores_column_order(real_msno):
    """A reordered table must come back in the trained order."""
    row = real_service.state["features"].loc[[real_msno]].drop(
        columns=["is_churn"], errors="ignore")
    shuffled = row.loc[:, list(reversed(row.columns))]
    assert list(real_service.align(shuffled).columns) == \
        real_service.state["feature_columns"]


def test_label_column_is_never_fed_to_the_model(real_msno):
    assert "is_churn" not in real_service.state["feature_columns"]


# ------------------------------------------------------- route isolation

def test_real_and_synthetic_states_are_separate():
    """Neither service may share model or feature-table objects with the other."""
    from src.api.main import _state
    assert real_service.state is not _state
    if _state["model"] is not None and real_service.state["model"] is not None:
        assert real_service.state["model"] is not _state["model"]
    if _state["features"] is not None and real_service.state["features"] is not None:
        assert real_service.state["features"] is not _state["features"]


def test_synthetic_route_still_serves_the_synthetic_model(api_client, sample_msno):
    """Stage 4 must not have disturbed the existing path."""
    res = api_client.post(f"/predict/{sample_msno}")
    assert res.status_code == 200
    body = res.json()
    assert body["msno"] == sample_msno
    # The synthetic response has no data_mode; that field is real-only.
    assert "data_mode" not in body


def test_a_synthetic_id_is_not_found_by_the_real_route(api_client, sample_msno):
    """Synthetic IDs (SYN...) are not KKBOX hashes, so the real route must 404
    rather than score one against the wrong feature space."""
    _skip_if_unavailable(api_client)
    res = api_client.post(f"/predict-real/{sample_msno}")
    assert res.status_code == 404


def test_a_real_id_is_not_found_by_the_synthetic_route(api_client, real_msno):
    res = api_client.post(f"/predict/{real_msno}")
    assert res.status_code == 404


def test_the_two_models_are_different_registered_names(api_client):
    _skip_if_unavailable(api_client)
    synthetic = api_client.get("/model-info").json()["model_name"]
    real = api_client.get("/health/real").json()["model_name"]
    assert synthetic == config.MLFLOW_MODEL_NAME == "ChurnCastModel"
    assert real == "ChurnCastRealV2"
    assert synthetic != real


# ------------------------------------------------- slash-safe id handling

SLASH_NOTE = ("KKBOX ids are base64; roughly half contain '/'. A path segment "
              "cannot carry one even percent-encoded, because Starlette decodes "
              "before routing.")


def test_ids_containing_a_slash_are_reachable(api_client):
    """Regression: the path-parameter form 404s for ~half the population."""
    _skip_if_unavailable(api_client)
    rows = api_client.get("/population/subscribers?limit=200")
    if rows.status_code != 200:
        pytest.skip("population artifact not built")
    with_slash = [r["msno"] for r in rows.json()["rows"] if "/" in r["msno"]]
    if not with_slash:
        pytest.skip("no slash-containing ids in this sample")

    msno = with_slash[0]
    res = api_client.get("/subscriber-real/analysis", params={"msno": msno})
    assert res.status_code == 200, SLASH_NOTE
    assert res.json()["msno"] == msno

    pred = api_client.post("/predict-real", params={"msno": msno})
    assert pred.status_code == 200
    assert 0.0 <= pred.json()["churn_probability"] <= 1.0


def test_query_and_path_forms_agree_for_ids_without_a_slash(api_client):
    _skip_if_unavailable(api_client)
    msno = api_client.get("/subscribers-real/sample?n=1").json()["subscribers"][0]
    if "/" in msno:
        pytest.skip("sample id contains a slash")
    a = api_client.post(f"/predict-real/{msno}").json()
    b = api_client.post("/predict-real", params={"msno": msno}).json()
    assert a["churn_probability"] == b["churn_probability"]


def test_query_form_rejects_a_blank_id(api_client):
    _skip_if_unavailable(api_client)
    assert api_client.get("/subscriber-real/analysis",
                          params={"msno": "   "}).status_code == 422
