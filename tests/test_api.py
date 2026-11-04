import io

import pytest

from api.server import create_api
from visionguard.inference import MockPredictor


@pytest.fixture
def api(monkeypatch):
    monkeypatch.setenv("API_KEY", "secret123")
    return create_api(MockPredictor()).test_client()


def test_health_public(api):
    r = api.get("/api/v1/health")
    assert r.status_code == 200 and r.json["model_name"] == "mock"


def test_requires_key(api, png_bytes):
    r = api.post("/api/v1/predict", data={"image": (io.BytesIO(png_bytes), "a.png")})
    assert r.status_code == 401


def test_predict_with_key(api, png_bytes):
    r = api.post("/api/v1/predict", data={"image": (io.BytesIO(png_bytes), "a.png")},
                 headers={"X-API-Key": "secret123"})
    assert r.status_code == 200
    assert set(r.json) >= {"label", "confidence", "prob_ai", "threshold", "model_name", "model_version"}


def test_api_rejects_bad_file(api):
    r = api.post("/api/v1/predict", data={"image": (io.BytesIO(b"xx"), "a.jpg")},
                 headers={"X-API-Key": "secret123"})
    assert r.status_code == 422
