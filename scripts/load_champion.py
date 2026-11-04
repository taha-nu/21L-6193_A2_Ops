"""Task 3.6 - load the registered model by ALIAS and classify one image.

    python scripts/load_champion.py path/to/image.jpg [--alias champion]

The alias is resolved by the registry at load time, so promoting a new version to
`champion` changes what this script loads without touching the code.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.common import load_params, setup_mlflow  # noqa: E402
from visionguard.inference import load_image_bytes  # noqa: E402
from visionguard.preprocess import decode_and_resize, to_rgb_jpeg_bytes  # noqa: E402


def main():
    a = argparse.ArgumentParser()
    a.add_argument("image")
    a.add_argument("--alias", default="champion")
    args = a.parse_args()

    mlflow = setup_mlflow()
    name = load_params()["registry"]["model_name"]
    client = mlflow.MlflowClient()
    mv = client.get_model_version_by_alias(name, args.alias)
    uri = f"models:/{name}@{args.alias}"
    print(f"Loading {uri}  -> version {mv.version} (run {mv.run_id})")
    model = mlflow.tensorflow.load_model(uri)

    # threshold + image size chosen during training travel with the run
    import json

    cfg = json.loads(open(mlflow.artifacts.download_artifacts(
        run_id=mv.run_id, artifact_path="keras_model/serving_config.json")).read())
    import tensorflow as tf

    img = load_image_bytes(Path(args.image).read_bytes())
    x = decode_and_resize(tf.constant(to_rgb_jpeg_bytes(img)), cfg["img_size"])
    prob = float(model(tf.expand_dims(x, 0), training=False).numpy().ravel()[0])
    label = "AI-generated" if prob >= cfg["threshold"] else "Real"
    print(f"{args.image}: {label}  P(AI)={prob:.4f}  threshold={cfg['threshold']:.3f}")


if __name__ == "__main__":
    main()
