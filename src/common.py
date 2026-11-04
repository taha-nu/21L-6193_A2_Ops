"""Shared helpers: paths, params, seeds, MLflow/DagsHub connection, reproducibility tags."""
from __future__ import annotations

import os
import random
import subprocess
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"            # unzipped Kaggle dataset (DVC tracked)
PROC_DIR = ROOT / "data" / "processed"     # cleaned RGB JPEGs (DVC tracked)
SPLIT_DIR = ROOT / "data" / "splits"       # train/val/test CSV lists (git tracked, small)
REPORTS = ROOT / "reports"
FIG_DIR = REPORTS / "figures"
MODELS_DIR = ROOT / "models"

CLASSES = ["real", "ai"]                   # label index: real=0, ai=1  (sigmoid = P(ai))

try:  # load .env for local runs (never committed)
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
except ImportError:
    pass


def project_name() -> str:
    """Assignment rule: '<RollNumber>_A2_Ops' everywhere."""
    return os.getenv("PROJECT_NAME", f"{os.getenv('ROLL_NO', '21L-6193')}_A2_Ops")


def class_from_folder(path: Path) -> str | None:
    """The label is the folder the image sits in. Kaggle uses AiArtData/ and RealArt/."""
    for part in reversed(path.parts[:-1]):
        p = part.lower()
        if p.startswith("real"):
            return "real"
        if p.startswith("ai") or "aiart" in p or "fake" in p or "generated" in p:
            return "ai"
    return None


def load_params() -> dict:
    with open(ROOT / "params.yaml") as f:
        return yaml.safe_load(f)


def set_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    import numpy as np

    np.random.seed(seed)
    try:
        import keras

        keras.utils.set_random_seed(seed)  # python + numpy + tensorflow
    except ImportError:
        pass


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def dvc_data_version() -> str:
    """md5 of the dataset directory recorded in data/processed.dvc (the DVC pointer file)."""
    for name in ("processed.dvc", "raw.dvc"):
        f = ROOT / "data" / name
        if f.exists():
            outs = yaml.safe_load(f.read_text()).get("outs", [{}])
            return f"{name}:{outs[0].get('md5', 'unknown')}"
    return "untracked"


def setup_mlflow(experiment: str | None = None):
    """Connect to DagsHub if MLFLOW_TRACKING_URI is set in the environment, else ./mlruns.

    Credentials come only from env vars (MLFLOW_TRACKING_USERNAME / _PASSWORD), loaded
    from the git-ignored .env file. They are never written in code.
    """
    import mlflow

    uri = os.getenv("MLFLOW_TRACKING_URI") or (ROOT / "mlruns").as_uri()
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment or project_name())
    return mlflow
