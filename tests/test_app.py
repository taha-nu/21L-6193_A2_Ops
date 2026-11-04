import io

from app.models import PredictionLog


def post(client, data: bytes, name: str):
    return client.post("/predict", data={"image": (io.BytesIO(data), name)},
                       content_type="multipart/form-data")


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json["checks"]["database"] == "ok"


def test_upload_page(client):
    assert b"Analyse image" in client.get("/").data


def test_predict_logs_to_db(client, app, png_bytes):
    r = post(client, png_bytes, "../../etc/my photo.png")
    assert r.status_code == 200
    assert b"confident" in r.data
    with app.app_context():
        row = PredictionLog.query.one()
        assert row.filename == "etc_my_photo.png"          # sanitized
        assert row.model_name == "mock" and 0 <= row.confidence <= 1
    assert b"etc_my_photo.png" in client.get("/history").data
    assert client.get("/api/history").json[0]["label"] in ("AI-generated", "Real")


def test_wrong_type_rejected(client):
    r = post(client, b"%PDF-1.4 hello", "report.pdf")
    assert r.status_code == 415
    assert b"not allowed" in r.data


def test_undecodable_image_rejected(client):
    r = post(client, b"this is not an image at all", "fake.jpg")
    assert r.status_code == 422
    assert b"could not be decoded" in r.data


def test_too_large_rejected(client):
    r = post(client, b"\xff" * (2 * 1024 * 1024), "big.jpg")   # limit is 1 MB in tests
    assert r.status_code == 413


def test_missing_file(client):
    assert client.post("/predict", data={}).status_code == 400


def test_failures_not_logged(client, app):
    post(client, b"nope", "x.jpg")
    with app.app_context():
        assert PredictionLog.query.count() == 0


def test_confidence_is_threshold_relative():
    from visionguard.inference import confidence

    assert confidence(0.5, 0.5) == 0.5 and confidence(1.0, 0.5) == 1.0 and confidence(0.0, 0.5) == 1.0
    assert confidence(0.6, 0.6) == 0.5 and confidence(0.3, 0.6) == 0.75
