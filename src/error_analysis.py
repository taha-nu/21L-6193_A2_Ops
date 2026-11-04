"""Task 2.6 - show 10 misclassified test images with measurable clues for a hypothesis.

    python -m src.error_analysis --run-id <mlflow run id>   # uses its test_predictions.csv

For each mistake we print size, aspect ratio, a sharpness score (variance of the
Laplacian: low = blurry), colour saturation and JPEG-ness of the source, which are the
usual suspects (blur, painting style, heavy compression, tiny images, watermarks).
Write your own hypothesis per image in the report using these numbers + your eyes.
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image, ImageFilter  # noqa: E402

from src.common import FIG_DIR, PROC_DIR, REPORTS, setup_mlflow  # noqa: E402


def clues(path) -> dict:
    with Image.open(path) as im:
        g = np.asarray(im.convert("L").filter(ImageFilter.FIND_EDGES), dtype=float)
        hsv = np.asarray(im.convert("HSV"), dtype=float)
    return {"sharpness": round(g.var(), 1), "saturation": round(hsv[..., 1].mean() / 255, 2)}


def main():
    a = argparse.ArgumentParser()
    a.add_argument("--run-id", required=True)
    a.add_argument("--n", type=int, default=10)
    args = a.parse_args()
    mlflow = setup_mlflow()
    csv = mlflow.artifacts.download_artifacts(run_id=args.run_id, artifact_path="reports/test_predictions.csv")
    df = pd.read_csv(csv)
    wrong = df[df.pred != df.label].copy()
    wrong["error"] = np.where(wrong.label == 0, "real -> called AI (FP)", "AI -> called real (FN)")
    wrong["margin"] = (wrong.prob_ai - 0.5).abs()
    wrong = wrong.sort_values("margin", ascending=False).head(args.n)  # most confident mistakes first
    if wrong.empty:
        print("No misclassified test images.")
        return
    extra = pd.DataFrame([clues(PROC_DIR / f) for f in wrong.filepath], index=wrong.index)
    wrong = wrong.join(extra).assign(aspect=lambda d: (d.width / d.height).round(2))

    cols = 5
    rows = int(np.ceil(len(wrong) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.6 * rows))
    for ax, (_, r) in zip(np.ravel(axes), wrong.iterrows(), strict=False):
        with Image.open(PROC_DIR / r.filepath) as im:
            ax.imshow(im.convert("RGB"))
        ax.set_title(f"{r.error}\nP(AI)={r.prob_ai:.2f} {r.width}x{r.height}\nsharp={r.sharpness} sat={r.saturation}",
                     fontsize=7.5)
    for ax in np.ravel(axes):
        ax.axis("off")
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "misclassified.png", dpi=120)
    cols_out = ["filepath", "source", "error", "prob_ai", "width", "height", "aspect", "sharpness", "saturation"]
    wrong[cols_out].to_csv(REPORTS / "misclassified.csv", index=False)
    print(wrong[cols_out].to_markdown(index=False))
    print("Median sharpness of ALL test images:",
          round(float(np.median([clues(PROC_DIR / f)["sharpness"] for f in df.filepath])), 1))
    print("saved", FIG_DIR / "misclassified.png")


if __name__ == "__main__":
    main()
