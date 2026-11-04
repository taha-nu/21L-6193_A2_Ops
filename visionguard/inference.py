"""Inference module (Task 4.1).

The Keras model is loaded ONCE when the process starts (Predictor() is created in the
app factory). Loading a .keras file takes seconds and hundreds of MB of RAM; doing it on
every request would make each request slow and could exhaust memory under load.

Preprocessing reuses visionguard.preprocess, the exact code used to build the training
tf.data pipeline, so serving inputs match training inputs bit for bit.
"""
from __future__ import annotations

import io
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from visionguard.preprocess import IMG_SIZE, decode_and_resize, to_rgb_jpeg_bytes

LABEL_POSITIVE = "AI-generated"  # sigmoid output = P(AI-generated)
LABEL_NEGATIVE = "Real"
CAVEAT = (
    "This is a statistical estimate, not proof. Edited, compressed, screenshotted or "
    "very small images, and paintings or renders, are often misjudged. Do not use this "
    "result alone to accuse anyone."
)


@dataclass
class Prediction:
    label: str
    confidence: float      # confidence in the returned label, 0..1
    prob_ai: float         # raw model output P(AI)
    threshold: float
    model_name: str
    model_version: str
    latency_ms: float

    def to_dict(self) -> dict:
        return self.__dict__.copy()


def confidence(prob: float, thr: float) -> float:
    """Confidence in the returned label, measured from the decision threshold.

    0.5 = exactly on the threshold (a coin flip), 1.0 = as far from it as possible.
    With thr=0.5 this equals max(p, 1-p). With a shifted threshold (we use ~0.6 to avoid
    false accusations) the plain 1-p would show e.g. "Real, 45% confident", which is
    confusing, so we rescale each side of the threshold to [0.5, 1].
    """
    thr = min(max(thr, 1e-6), 1 - 1e-6)
    if prob >= thr:
        return 0.5 + 0.5 * (prob - thr) / (1 - thr)
    return 0.5 + 0.5 * (thr - prob) / thr


def load_image_bytes(raw: bytes) -> Image.Image:
    """Decode + fully load an image; raises on anything that is not a real image."""
    img = Image.open(io.BytesIO(raw))
    img.verify()                       # cheap structural check
    img = Image.open(io.BytesIO(raw))  # verify() invalidates the object; reopen
    img.load()                         # force full decode (catches truncated files)
    if getattr(img, "n_frames", 1) > 1:
        img.seek(0)                    # animated GIF/WebP: first frame only
    return img


class Predictor:
    def __init__(self, model_path: str | None = None, threshold: float | None = None):
        import keras

        self.model_path = Path(model_path or os.getenv("MODEL_PATH", "models/champion.keras"))
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file not found at {self.model_path}. Run scripts/fetch_model.py "
                "or mount the models/ folder."
            )
        self.model = keras.models.load_model(self.model_path, compile=False)

        # serving_config.json is written by fetch_model.py next to the model
        cfg_path = self.model_path.with_name("serving_config.json")
        cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        env_thr = os.getenv("THRESHOLD")
        self.threshold = float(threshold if threshold is not None else env_thr or cfg.get("threshold", 0.5))
        self.model_name = os.getenv("MODEL_NAME", cfg.get("model_name", self.model_path.stem))
        self.model_version = str(os.getenv("MODEL_VERSION", cfg.get("model_version", "local")))
        self.img_size = int(cfg.get("img_size", IMG_SIZE))
        # warm-up: first call builds the TF graph, so do it before real traffic
        self.predict_image(Image.new("RGB", (self.img_size, self.img_size)))

    def predict_image(self, img: Image.Image) -> Prediction:
        import tensorflow as tf

        t0 = time.perf_counter()
        x = decode_and_resize(tf.constant(to_rgb_jpeg_bytes(img)), self.img_size)
        prob = float(self.model(tf.expand_dims(x, 0), training=False).numpy().ravel()[0])
        is_ai = prob >= self.threshold
        return Prediction(
            label=LABEL_POSITIVE if is_ai else LABEL_NEGATIVE,
            confidence=round(confidence(prob, self.threshold), 4),
            prob_ai=round(prob, 4),
            threshold=round(self.threshold, 4),
            model_name=self.model_name,
            model_version=self.model_version,
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
        )

    def predict_bytes(self, raw: bytes) -> Prediction:
        return self.predict_image(load_image_bytes(raw))


class MockPredictor:
    """Used by tests/CI so they run without a trained model."""

    model_name, model_version, threshold = "mock", "0", 0.5

    def predict_bytes(self, raw: bytes) -> Prediction:
        img = load_image_bytes(raw)
        prob = (sum(img.convert("L").resize((8, 8)).getdata()) / 64) / 255
        is_ai = prob >= 0.5
        return Prediction(LABEL_POSITIVE if is_ai else LABEL_NEGATIVE,
                          round(confidence(prob, 0.5), 4), round(prob, 4), 0.5,
                          "mock", "0", 0.1)


if __name__ == "__main__":  # python -m visionguard.inference img1.jpg img2.jpg
    import sys

    p = Predictor()
    print(f"Loaded {p.model_name} v{p.model_version} threshold={p.threshold}")
    for f in sys.argv[1:]:
        print(f, "->", p.predict_bytes(Path(f).read_bytes()).to_dict())
