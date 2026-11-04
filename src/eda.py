"""Task 1.3 - Visual EDA. Needs reports/audit.csv from src.audit.

    python -m src.eda   # -> reports/figures/sample_grid.png, hist_*.png
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

from src.common import FIG_DIR, RAW_DIR, REPORTS  # noqa: E402

COLORS = {"real": "#2a7ab9", "ai": "#d9622b"}


def sample_grid(df: pd.DataFrame, n: int = 8, seed: int = 0) -> None:
    classes = sorted(df["class"].dropna().unique())
    fig, axes = plt.subplots(len(classes), n, figsize=(2 * n, 2.3 * len(classes)))
    for r, c in enumerate(classes):
        rows = df[df["class"] == c].sample(min(n, (df["class"] == c).sum()), random_state=seed)
        for k, ax in enumerate(np.atleast_2d(axes)[r]):
            ax.axis("off")
            if k < len(rows):
                row = rows.iloc[k]
                with Image.open(RAW_DIR / row.path) as im:
                    im.thumbnail((256, 256))
                    ax.imshow(im.convert("RGB"))
                ax.set_title(f"{c} {row.width}x{row.height}", fontsize=8)
    fig.suptitle("Random samples per class")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "sample_grid.png", dpi=120)
    plt.close(fig)


def hist(df: pd.DataFrame, col: str, xlabel: str, scale: float = 1.0, log: bool = False) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    vals_all = df[col] / scale
    bins = (np.logspace(np.log10(max(vals_all.min(), 1)), np.log10(vals_all.max()), 40)
            if log else np.linspace(vals_all.min(), vals_all.quantile(0.99), 40))
    for c, g in df.groupby("class"):
        ax.hist(g[col] / scale, bins=bins, alpha=0.55, label=f"{c} (n={len(g)})", color=COLORS.get(c))
        ax.axvline((g[col] / scale).median(), color=COLORS.get(c), ls="--", lw=1)
    if log:
        ax.set_xscale("log")
    ax.set(xlabel=xlabel, ylabel="images", title=f"{xlabel} by class (dashed = median)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"hist_{col}.png", dpi=120)
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(REPORTS / "audit.csv")
    df = df[df.readable & ~df.duplicate]
    sample_grid(df)
    hist(df, "width", "Width (px)")
    hist(df, "height", "Height (px)")
    hist(df, "file_size", "File size (KB)", scale=1024, log=True)
    # aspect ratio is useful when explaining the resize choice in Task 2.1
    df = df.assign(aspect=df.width / df.height)
    hist(df, "aspect", "Aspect ratio (w/h)")
    print("Saved figures to", FIG_DIR)
    print(df.groupby("class")[["width", "height", "file_size", "aspect"]].describe().T.round(2))


if __name__ == "__main__":
    main()
