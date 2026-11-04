"""Task 5 - the database trail. One row per prediction: what was decided, when, by which
model, and how confidently. The image itself is NOT stored, only its SHA-256, so the
same image can be recognised without keeping user content."""
from datetime import datetime, timezone

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class PredictionLog(db.Model):
    __tablename__ = "predictions"

    id = db.Column(db.Integer, primary_key=True)
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, index=True,
                           default=lambda: datetime.now(timezone.utc))
    filename = db.Column(db.String(255), nullable=False)      # sanitized name
    image_sha256 = db.Column(db.String(64), nullable=False, index=True)
    image_width = db.Column(db.Integer)
    image_height = db.Column(db.Integer)
    label = db.Column(db.String(32), nullable=False)          # decision
    confidence = db.Column(db.Float, nullable=False)          # confidence in that decision
    prob_ai = db.Column(db.Float, nullable=False)             # raw model output
    threshold = db.Column(db.Float, nullable=False)
    model_name = db.Column(db.String(64), nullable=False)
    model_version = db.Column(db.String(32), nullable=False)
    inference_mode = db.Column(db.String(16), nullable=False) # local | remote
    latency_ms = db.Column(db.Float)
    client_ip = db.Column(db.String(64))

    def to_dict(self):
        d = {c.name: getattr(self, c.name) for c in self.__table__.columns}
        d["created_at"] = self.created_at.isoformat()
        return d
