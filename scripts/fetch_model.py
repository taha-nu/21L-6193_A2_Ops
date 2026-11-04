"""Download the champion model ONCE from DagsHub/MLflow into models/ (deployment step).

The serving app never talks to DagsHub per request: this script runs at deploy/build
time and writes
    models/champion.keras
    models/serving_config.json   (threshold, image size, model name + version)
The web container mounts models/ read-only; the API image copies it in at build time.

    python scripts/fetch_model.py [--alias champion] [--out models]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import load_params, setup_mlflow  # noqa: E402


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--alias", default="champion")
    a.add_argument("--out", default="models")
    args = a.parse_args()

    mlflow = setup_mlflow()
    name = load_params()["registry"]["model_name"]
    mv = mlflow.MlflowClient().get_model_version_by_alias(name, args.alias)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    local = Path(mlflow.artifacts.download_artifacts(run_id=mv.run_id, artifact_path="keras_model"))
    shutil.copy(local / "model.keras", out / "champion.keras")
    cfg = json.loads((local / "serving_config.json").read_text())
    cfg.update(model_name=name, model_version=str(mv.version), alias=args.alias)
    (out / "serving_config.json").write_text(json.dumps(cfg, indent=2))
    print(f"Saved {name} v{mv.version} ({args.alias}) to {out / 'champion.keras'}  threshold={cfg['threshold']:.3f}")


if __name__ == "__main__":
    main()
