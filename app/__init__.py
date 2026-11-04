"""VisionGuard web app (Tasks 4, 5, 8).

All settings come from environment variables (Task 4.4), so the same code runs on a
laptop, in Docker Compose and in the cloud without edits, and no secret lives in Git:

  SECRET_KEY       Flask session/flash signing key
  DATABASE_URL     postgresql+psycopg://user:pass@db:5432/visionguard  (default: local SQLite)
  INFERENCE_MODE   local  -> load MODEL_PATH in this process
                   remote -> call the cloud model API at API_URL with API_KEY  (Task 8)
                   mock   -> fake predictor for tests/CI
  MODEL_PATH       models/champion.keras        THRESHOLD   overrides serving_config.json
  MAX_UPLOAD_MB    8
"""
from __future__ import annotations

import logging
import os
import time

from flask import Flask
from sqlalchemy import text

from app.models import db

log = logging.getLogger("visionguard")


def make_predictor(mode: str):
    if mode == "mock":
        from visionguard.inference import MockPredictor

        return MockPredictor()
    if mode == "remote":
        from app.remote import RemotePredictor

        p = RemotePredictor(os.getenv("API_URL", ""), os.getenv("API_KEY", ""))
        try:
            log.info("Model API health: %s", p.health())
        except Exception as e:  # noqa: BLE001 - app still starts; /health will show it
            log.warning("Model API not reachable at start-up: %s", e)
        return p
    from visionguard.inference import Predictor

    return Predictor()  # loads the Keras model ONCE, here, at start-up


def create_app(overrides: dict | None = None) -> Flask:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY=os.getenv("SECRET_KEY", "dev-only-change-me"),
        SQLALCHEMY_DATABASE_URI=os.getenv("DATABASE_URL", "sqlite:///visionguard.db"),
        SQLALCHEMY_ENGINE_OPTIONS={"pool_pre_ping": True},
        MAX_UPLOAD_MB=float(os.getenv("MAX_UPLOAD_MB", "8")),
        INFERENCE_MODE=os.getenv("INFERENCE_MODE", "local"),
    )
    if overrides:
        app.config.update(overrides)
    # Werkzeug rejects bigger request bodies with 413 before our code even runs
    app.config["MAX_CONTENT_LENGTH"] = int(app.config["MAX_UPLOAD_MB"] * 1024 * 1024) + 64 * 1024

    db.init_app(app)
    with app.app_context():
        for attempt in range(10):   # the DB may still be starting (Compose also waits on its healthcheck)
            try:
                db.session.execute(text("SELECT 1"))
                db.create_all()
                break
            except Exception as e:  # noqa: BLE001
                log.warning("Database not ready (%s), retry %d/10", type(e).__name__, attempt + 1)
                db.session.rollback()
                time.sleep(3)
        else:
            raise RuntimeError("Could not connect to the database at DATABASE_URL")
        db_url = db.engine.url.render_as_string(hide_password=True)

    app.extensions["predictor"] = make_predictor(app.config["INFERENCE_MODE"])
    log.info("VisionGuard ready: mode=%s db=%s", app.config["INFERENCE_MODE"], db_url)

    from app.routes import bp, register_error_handlers

    app.register_blueprint(bp)
    register_error_handlers(app)
    return app
