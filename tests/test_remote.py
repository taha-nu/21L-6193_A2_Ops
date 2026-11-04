"""Web app in remote mode talking to the API (both in-process, requests patched)."""
import io

import pytest

from api.server import create_api
from visionguard.inference import MockPredictor


class FakeResp:
    def __init__(self, r):
        self.status_code, self._json, self.text = r.status_code, r.get_json(), r.get_data(as_text=True)

    def json(self):
        return self._json

    def raise_for_status(self):
        assert self.status_code < 400


@pytest.fixture
def remote_client(tmp_path, monkeypatch):
    monkeypatch.setenv("API_KEY", "k")
    api = create_api(MockPredictor()).test_client()

    def fake_get(self, url, **kw):
        return FakeResp(api.get(url.split("example.test")[1]))

    def fake_post(self, url, files=None, **kw):
        name, raw = files["image"]
        return FakeResp(api.post(url.split("example.test")[1], data={"image": (io.BytesIO(raw), name)},
                                 headers=dict(self.headers)))

    monkeypatch.setattr("requests.Session.get", fake_get)
    monkeypatch.setattr("requests.Session.post", fake_post)
    monkeypatch.setenv("API_URL", "https://example.test")
    from app import create_app

    return create_app({"SQLALCHEMY_DATABASE_URI": f"sqlite:///{tmp_path / 'r.db'}",
                       "INFERENCE_MODE": "remote"}).test_client()


def test_remote_predict_and_history(remote_client, png_bytes):
    r = remote_client.post("/predict", data={"image": (io.BytesIO(png_bytes), "x.png")},
                           content_type="multipart/form-data")
    assert r.status_code == 200
    h = remote_client.get("/api/history").json
    assert h[0]["inference_mode"] == "remote" and h[0]["model_name"] == "mock"
    assert remote_client.get("/health").json["checks"]["model_api"] == "ok"
