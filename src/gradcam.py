"""Bonus - Grad-CAM for 4 correct and 4 wrong test predictions.

    python -m src.gradcam --run-id <id>        (or --model models/champion.keras)

Our models are plain layer chains, so we run them layer by layer, record the output of
the last convolutional stage (last Conv2D in the baseline, the whole backbone in the
transfer model) and take the gradient of P(AI) with respect to it. The heat map is the
ReLU of the gradient-weighted sum of feature maps: bright = pixels that pushed the
prediction towards "AI".
"""
from __future__ import annotations

import argparse
import json

import keras
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tensorflow as tf  # noqa: E402

from src.common import FIG_DIR, PROC_DIR, setup_mlflow  # noqa: E402
from visionguard.preprocess import decode_and_resize  # noqa: E402


def target_index(model) -> int:
    """Index of the last conv-stage output before global pooling (backbone or last ReLU)."""
    idx = None
    for i, l in enumerate(model.layers):
        if isinstance(l, (keras.Model, keras.layers.Conv2D, keras.layers.ReLU)):
            idx = i
        if isinstance(l, keras.layers.GlobalAveragePooling2D):
            break
    return idx


def gradcam(model, x, idx):
    with tf.GradientTape() as tape:
        h = x
        conv = None
        for i, l in enumerate(model.layers[1:], start=1):
            h = l(h, training=False)
            if i == idx:
                conv = h
                tape.watch(conv)
        prob = h[:, 0]
    grads = tape.gradient(prob, conv)
    w = tf.reduce_mean(grads, axis=(1, 2), keepdims=True)
    cam = tf.nn.relu(tf.reduce_sum(w * conv, axis=-1))[0].numpy()
    return cam / (cam.max() + 1e-8), float(prob[0])


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--run-id")
    a.add_argument("--model", default="models/champion.keras")
    args = a.parse_args()
    if args.run_id:
        mlflow = setup_mlflow()
        mdir = mlflow.artifacts.download_artifacts(run_id=args.run_id, artifact_path="keras_model")
        preds = pd.read_csv(mlflow.artifacts.download_artifacts(
            run_id=args.run_id, artifact_path="reports/test_predictions.csv"))
        model_path, cfg = f"{mdir}/model.keras", json.load(open(f"{mdir}/serving_config.json"))
    else:
        raise SystemExit("use --run-id so the matching test predictions are available")
    model = keras.models.load_model(model_path)
    idx = target_index(model)
    print("Grad-CAM target layer:", model.layers[idx].name)

    right = preds[preds.pred == preds.label].sample(min(4, (preds.pred == preds.label).sum()), random_state=0)
    wrong = preds[preds.pred != preds.label].head(4)
    sel = pd.concat([right.assign(kind="correct"), wrong.assign(kind="wrong")])
    fig, axes = plt.subplots(2, len(sel), figsize=(2.6 * len(sel), 5.4))
    axes = np.array(axes).reshape(2, -1)
    for k, (_, r) in enumerate(sel.iterrows()):
        x = decode_and_resize(tf.io.read_file(str(PROC_DIR / r.filepath)), cfg["img_size"])[None]
        cam, p = gradcam(model, x, idx)
        cam = tf.image.resize(cam[..., None], x.shape[1:3]).numpy()[..., 0]
        axes[0, k].imshow(x[0].numpy().astype("uint8"))
        axes[0, k].set_title(f"{r.kind}\ntrue={'ai' if r.label else 'real'} P(AI)={p:.2f}", fontsize=8)
        axes[1, k].imshow(x[0].numpy().astype("uint8"))
        axes[1, k].imshow(cam, cmap="jet", alpha=0.45)
    for ax in axes.ravel():
        ax.axis("off")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "gradcam.png", dpi=120)
    print("saved", FIG_DIR / "gradcam.png")


if __name__ == "__main__":
    main()
