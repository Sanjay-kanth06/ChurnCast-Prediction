"""Shared fixtures for the ChurnCast test suite."""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)


@pytest.fixture(scope="session")
def feature_table():
    """The persisted feature table produced by the pipeline."""
    from src.features import load_feature_table
    try:
        return load_feature_table()
    except FileNotFoundError:
        pytest.skip("feature table not built; run scripts/run_pipeline.py first")


@pytest.fixture(scope="session")
def raw_tables():
    from src.features.features import load_raw
    from src import config
    if not config.RAW_FILES["train"].exists():
        pytest.skip("raw data not generated; run scripts/generate_sample_data.py first")
    return load_raw()


@pytest.fixture(scope="session")
def api_client():
    """TestClient with the app lifespan running, so the model is loaded."""
    from fastapi.testclient import TestClient
    from src.api.main import app
    with TestClient(app) as client:
        yield client


@pytest.fixture(scope="session")
def sample_msno(feature_table):
    return str(feature_table.index[0])
