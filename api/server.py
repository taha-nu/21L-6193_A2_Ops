"""Task 7 - the model API that runs in the cloud.

  GET  /api/v1/health    public: model name/version/threshold (no secrets)
  POST /api/v1/predict   multipart field "image"; header  X-API-Key: <API_KEY>

The model is loaded once at start-up. The key is compared with hmac.compare_digest
(constant time, so response timing does not leak how many characters matched). The API
keeps no state: the history database lives with the Flask UI.

Run:  gunicorn -b 0.0.0.0:8080 -w 1 --threads 4 "api.server:create_api()"
"""
from __future__ import annotations

import hmac
import logging
import os

from flask import Flask, jsonify, request
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from visionguard.validation import UploadError, validate_upload

log = logging.getLogger("visionguard.api")


def create_api(predictor=None) -> Flask:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = Flask(__name__)
    max_mb = float(os.getenv("MAX_UPLOAD_MB", "8"))
    app.config["MAX_CONTENT_LENGTH"] = int(max_mb * 1024 * 1024) + 64 * 1024
    api_key = os.getenv("API_KEY", "")
    if not api_key:
        log.warning("API_KEY is empty: /api/v1/predict is UNPROTECTED")

    if predictor is None:
        if os.getenv("INFERENCE_MODE") == "mock":
            from visionguard.inference import MockPredictor
            predictor = MockPredictor()
        else:
            from visionguard.inference import Predictor
            predictor = Predictor()
    log.info("Model API ready: %s v%s threshold=%s", predictor.model_name,
             predictor.model_version, predictor.threshold)

    @app.get("/")
    @app.get("/api/v1/health")
    def health():
        return jsonify(status="ok", service="visionguard-model-api", model_name=predictor.model_name,
                       model_version=str(predictor.model_version), threshold=predictor.threshold)

    @app.post("/api/v1/predict")
    def predict():
        if api_key and not hmac.compare_digest(request.headers.get("X-API-Key", ""), api_key):
            return jsonify(error="invalid or missing X-API-Key"), 401
        f = request.files.get("image")
        raw = f.read() if f else b""
        name, img = validate_upload(f.filename if f else None, raw, max_mb)
        pred = predictor.predict_image(img) if hasattr(predictor, "predict_image") \
            else predictor.predict_bytes(raw)
        log.info("predict file=%s label=%s prob_ai=%.4f", name, pred.label, pred.prob_ai)
        return jsonify(pred.to_dict() | {"filename": name, "width": img.width, "height": img.height})

    @app.errorhandler(UploadError)
    def bad_upload(e):
        return jsonify(error=e.message, title=e.title), e.status

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(e):
        return jsonify(error=f"file larger than {max_mb:g} MB"), 413

    @app.errorhandler(HTTPException)
    def http_err(e):
        return jsonify(error=e.description), e.code

    @app.errorhandler(Exception)
    def boom(e):
        log.exception("unhandled")
        return jsonify(error="internal error"), 500

    return app
