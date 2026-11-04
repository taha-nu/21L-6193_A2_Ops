"""Pages: upload (/), result (POST /predict), history (/history), health (/health),
plus a JSON history API (/api/history) handy for the report."""
from __future__ import annotations

import hashlib
import logging

from flask import (Blueprint, current_app, jsonify, render_template, request)
from sqlalchemy import func, text
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from app.models import PredictionLog, db
from visionguard.inference import CAVEAT
from visionguard.validation import ALLOWED_EXT, UploadError, validate_upload

bp = Blueprint("main", __name__)
log = logging.getLogger("visionguard")


def predictor():
    return current_app.extensions["predictor"]


@bp.get("/")
def index():
    return render_template("index.html", allowed=sorted(ALLOWED_EXT),
                           max_mb=current_app.config["MAX_UPLOAD_MB"])


@bp.post("/predict")
def predict():
    f = request.files.get("image")
    raw = f.read() if f else b""          # bytes in memory only; never saved to disk
    safe_name, img = validate_upload(f.filename if f else None, raw, current_app.config["MAX_UPLOAD_MB"])

    mode = current_app.config["INFERENCE_MODE"]
    p = predictor()
    if mode == "remote":
        pred = p.predict_bytes(raw, safe_name)      # cloud model API (Task 8)
    elif hasattr(p, "predict_image"):
        pred = p.predict_image(img)                 # local model, image already decoded
    else:
        pred = p.predict_bytes(raw)                 # mock

    row = PredictionLog(filename=safe_name, image_sha256=hashlib.sha256(raw).hexdigest(),
                        image_width=img.width, image_height=img.height, label=pred.label,
                        confidence=pred.confidence, prob_ai=pred.prob_ai, threshold=pred.threshold,
                        model_name=pred.model_name, model_version=str(pred.model_version),
                        inference_mode=mode, latency_ms=pred.latency_ms,
                        client_ip=request.headers.get("X-Forwarded-For", request.remote_addr))
    db.session.add(row)
    db.session.commit()
    log.info("prediction id=%s file=%s label=%s conf=%.3f model=%s v%s", row.id, safe_name,
             pred.label, pred.confidence, pred.model_name, pred.model_version)
    return render_template("result.html", p=pred, row=row, caveat=CAVEAT)


@bp.get("/history")
def history():
    page = request.args.get("page", 1, type=int)
    q = PredictionLog.query.order_by(PredictionLog.created_at.desc())
    rows = q.paginate(page=page, per_page=20, error_out=False)
    stats = db.session.query(PredictionLog.label, func.count(), func.avg(PredictionLog.confidence)) \
        .group_by(PredictionLog.label).all()
    return render_template("history.html", rows=rows, stats=stats)


@bp.get("/api/history")
def history_json():
    limit = min(request.args.get("limit", 50, type=int), 500)
    rows = PredictionLog.query.order_by(PredictionLog.created_at.desc()).limit(limit).all()
    return jsonify([r.to_dict() for r in rows])


@bp.get("/health")
def health():
    status, checks = 200, {}
    try:
        db.session.execute(text("SELECT 1"))
        checks["database"] = "ok"
    except Exception as e:  # noqa: BLE001
        checks["database"], status = f"error: {type(e).__name__}", 503
    p = predictor()
    mode = current_app.config["INFERENCE_MODE"]
    if mode == "remote":
        try:
            p.health()
            checks["model_api"] = "ok"
        except Exception as e:  # noqa: BLE001
            checks["model_api"], status = f"error: {type(e).__name__}", 503
    else:
        checks["model"] = "loaded"
    return jsonify(status="ok" if status == 200 else "degraded", mode=mode, model_name=p.model_name,
                   model_version=str(p.model_version), threshold=p.threshold, checks=checks), status


def register_error_handlers(app):
    def page(status, title, message):
        if request.path.startswith("/api/"):
            return jsonify(error=message), status
        return render_template("error.html", status=status, title=title, message=message), status

    @app.errorhandler(UploadError)
    def upload_error(e):
        log.warning("rejected upload: %s %s", e.status, e.message)
        return page(e.status, e.title, e.message)

    @app.errorhandler(RequestEntityTooLarge)
    def too_large(e):
        return page(413, "File too large", f"The limit is {app.config['MAX_UPLOAD_MB']:g} MB.")

    from app.remote import PredictionServiceError

    @app.errorhandler(PredictionServiceError)
    def api_down(e):
        log.error("model API error: %s", e)
        return page(502, "Model service unavailable", "The prediction service did not answer. Try again later.")

    @app.errorhandler(HTTPException)
    def http_error(e):
        return page(e.code, e.name, e.description)

    @app.errorhandler(Exception)
    def unexpected(e):
        log.exception("unhandled error")
        return page(500, "Something went wrong", "An unexpected error occurred. It has been logged.")
