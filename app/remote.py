"""Task 8 - client for the model API deployed in the cloud (api/server.py).

The local Flask UI sends the uploaded bytes to  POST {API_URL}/api/v1/predict  with the
secret key in the X-API-Key header, and gets back the same fields the local Predictor
returns. The key is read from the environment, never from code.
"""
from __future__ import annotations

import requests

from visionguard.inference import Prediction


class PredictionServiceError(Exception):
    pass


class RemotePredictor:
    def __init__(self, api_url: str, api_key: str, timeout: float = 30):
        if not api_url:
            raise ValueError("INFERENCE_MODE=remote needs API_URL")
        self.api_url = api_url.rstrip("/")
        self.session = requests.Session()           # re-use the TCP/TLS connection
        self.session.headers["X-API-Key"] = api_key or ""
        self.timeout = timeout
        self.model_name, self.model_version, self.threshold = "remote", "?", None

    def health(self) -> dict:
        r = self.session.get(f"{self.api_url}/api/v1/health", timeout=10)
        r.raise_for_status()
        info = r.json()
        self.model_name = info.get("model_name", self.model_name)
        self.model_version = str(info.get("model_version", self.model_version))
        self.threshold = info.get("threshold")
        return info

    def predict_bytes(self, raw: bytes, filename: str = "upload.jpg") -> Prediction:
        try:
            r = self.session.post(f"{self.api_url}/api/v1/predict",
                                  files={"image": (filename, raw)}, timeout=self.timeout)
        except requests.RequestException as e:
            raise PredictionServiceError(f"Model API unreachable: {type(e).__name__}") from e
        if r.status_code != 200:
            try:
                msg = r.json().get("error", r.text[:200])
            except ValueError:
                msg = r.text[:200]
            raise PredictionServiceError(f"Model API returned {r.status_code}: {msg}")
        d = r.json()
        keys = Prediction.__dataclass_fields__.keys()
        return Prediction(**{k: d[k] for k in keys})
