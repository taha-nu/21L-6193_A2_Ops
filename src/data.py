"""Task 2.1 - tf.data input pipeline.

Stages (in order):
  1. from_tensor_slices(paths, labels)  - one element per row of the split CSV
  2. shuffle (train only)               - reshuffled every epoch, fixed seed
  3. map(load)                          - read file -> decode JPEG -> resize_with_pad to
                                          IMG_SIZE (shared visionguard.preprocess code)
  4. cache (val/test only)              - decoded tensors kept in RAM after epoch 1
  5. batch(BATCH_SIZE)
  6. map(augment) (train only)          - random flip / rotation / zoom / contrast,
                                          applied on the batch, so each epoch sees new variants
  7. prefetch(AUTOTUNE)                 - CPU prepares batch n+1 while the model trains on n

Aspect ratio: resize_with_pad scales the longer side to IMG_SIZE and pads the shorter
side with black bars, so faces/objects are not stretched. Stretching would distort the
very textures and proportions that can reveal AI generation.
"""
from __future__ import annotations

import pandas as pd
import tensorflow as tf

from src.common import PROC_DIR, SPLIT_DIR
from visionguard.preprocess import decode_and_resize

AUTOTUNE = tf.data.AUTOTUNE


def read_split(name: str) -> pd.DataFrame:
    return pd.read_csv(SPLIT_DIR / f"{name}.csv")


def build_augmenter(cfg: dict, seed: int) -> tf.keras.Sequential:
    layers = []
    if cfg.get("flip"):
        layers.append(tf.keras.layers.RandomFlip("horizontal", seed=seed))
    if cfg.get("rotation"):
        layers.append(tf.keras.layers.RandomRotation(cfg["rotation"], fill_mode="constant", seed=seed))
    if cfg.get("zoom"):
        layers.append(tf.keras.layers.RandomZoom(cfg["zoom"], fill_mode="constant", seed=seed))
    if cfg.get("contrast"):
        layers.append(tf.keras.layers.RandomContrast(cfg["contrast"], value_range=(0, 255), seed=seed))
    return tf.keras.Sequential(layers, name="augment")


def make_dataset(split: str, img_size: int, batch_size: int, seed: int,
                 augment_cfg: dict | None = None, shuffle: bool | None = None) -> tf.data.Dataset:
    df = read_split(split)
    paths = [str(PROC_DIR / p) for p in df.filepath]
    labels = df.label.astype("float32").values
    is_train = split == "train"
    shuffle = is_train if shuffle is None else shuffle

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(df), seed=seed, reshuffle_each_iteration=True)

    def load(path, label):
        return decode_and_resize(tf.io.read_file(path), img_size), label

    ds = ds.map(load, num_parallel_calls=AUTOTUNE)
    if not is_train:
        ds = ds.cache()
    ds = ds.batch(batch_size)
    if is_train and augment_cfg and any(augment_cfg.values()):
        aug = build_augmenter(augment_cfg, seed)
        ds = ds.map(lambda x, y: (tf.clip_by_value(aug(x, training=True), 0, 255), y),
                    num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


def plot_augmented_batch(out_path, img_size=224, batch_size=16, seed=42, augment_cfg=None):
    """Figure for the report: one augmented training batch."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ds = make_dataset("train", img_size, batch_size, seed, augment_cfg)
    x, y = next(iter(ds))
    n = min(16, len(y))
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i, ax in enumerate(axes.ravel()):
        ax.axis("off")
        if i < n:
            ax.imshow(x[i].numpy().astype("uint8"))
            ax.set_title("ai" if y[i] == 1 else "real", fontsize=9)
    fig.suptitle(f"One augmented training batch ({img_size}x{img_size}, padded)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    from src.common import FIG_DIR, load_params

    p = load_params()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plot_augmented_batch(FIG_DIR / "augmented_batch.png", p["img_size"], 16, p["seed"], p["augment"])
    print("saved", FIG_DIR / "augmented_batch.png")
