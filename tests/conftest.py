import io

import pytest
from PIL import Image


@pytest.fixture
def png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), (200, 30, 30)).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("INFERENCE_MODE", "mock")
    from app import create_app

    return create_app({"SQLALCHEMY_DATABASE_URI": f"sqlite:///{tmp_path / 't.db'}",
                       "INFERENCE_MODE": "mock", "TESTING": True, "MAX_UPLOAD_MB": 1})


@pytest.fixture
def client(app):
    return app.test_client()
