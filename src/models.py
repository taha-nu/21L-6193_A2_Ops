"""Task 2.2 (baseline CNN) and 2.3 (transfer learning) model builders.

Both models take raw pixels in [0, 255]. Scaling is done by a layer INSIDE the model,
so the saved .keras file carries its own preprocessing and serving cannot get it wrong.
Both are simple layer chains, which also makes Grad-CAM easy (src/gradcam.py).
"""
from __future__ import annotations

import os

import keras
from keras import layers

# backbone -> (constructor, preprocessing layer, name prefix of its last block)
BACKBONES = {
    # MobileNetV2 expects pixels in [-1, 1]: x/127.5 - 1 (same as its preprocess_input)
    "mobilenet_v2": (keras.applications.MobileNetV2,
                     lambda: layers.Rescaling(1 / 127.5, offset=-1, name="mobilenet_preprocess"),
                     "block_16"),
    # EfficientNetB0 has its own Rescaling+Normalization inside and expects [0, 255]
    "efficientnet_b0": (keras.applications.EfficientNetB0,
                        lambda: layers.Identity(name="efficientnet_preprocess"),
                        "block7"),
}


def compile_binary(model: keras.Model, lr: float) -> keras.Model:
    """Binary cross-entropy matches one sigmoid output (it is the log-likelihood of a
    Bernoulli label). Adam adapts the step size per weight and works well out of the box.
    Accuracy alone hides class imbalance, so precision, recall and AUC are tracked too."""
    model.compile(
        optimizer=keras.optimizers.Adam(lr),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy"),
                 keras.metrics.Precision(name="precision"),
                 keras.metrics.Recall(name="recall"),
                 keras.metrics.AUC(name="auc")],
    )
    return model


def build_baseline(img_size: int = 224, dropout: float = 0.4) -> keras.Model:
    def block(x, filters, i):
        x = layers.Conv2D(filters, 3, padding="same", use_bias=False, name=f"conv{i}")(x)  # local patterns
        x = layers.BatchNormalization(name=f"bn{i}")(x)   # stable activations -> faster training
        x = layers.ReLU(name=f"relu{i}")(x)               # non-linearity
        return layers.MaxPooling2D(name=f"pool{i}")(x)    # halve resolution, grow receptive field

    inp = keras.Input((img_size, img_size, 3), name="image")
    x = layers.Rescaling(1 / 255.0, name="rescale")(inp)  # [0,255] -> [0,1]
    for i, f in enumerate([32, 64, 128, 256], start=1):
        x = block(x, f, i)
    x = layers.GlobalAveragePooling2D(name="gap")(x)       # one number per filter, few params
    x = layers.Dropout(dropout, name="dropout")(x)        # regularization against overfitting
    out = layers.Dense(1, activation="sigmoid", name="prob_ai")(x)  # P(AI-generated)
    return keras.Model(inp, out, name="baseline_cnn")


def build_transfer(backbone: str = "mobilenet_v2", img_size: int = 224, dropout: float = 0.3):
    ctor, prep, _ = BACKBONES[backbone]
    weights = None if os.getenv("VG_WEIGHTS", "imagenet") == "none" else "imagenet"  # offline tests
    base = ctor(include_top=False, weights=weights, input_shape=(img_size, img_size, 3))
    base.trainable = False                                # stage 1: frozen feature extractor

    inp = keras.Input((img_size, img_size, 3), name="image")
    x = prep()(inp)
    x = base(x, training=False)  # keep BatchNorm in inference mode, also after unfreezing
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(dropout, name="dropout")(x)
    out = layers.Dense(1, activation="sigmoid", name="prob_ai")(x)
    return keras.Model(inp, out, name=f"transfer_{backbone}"), base


def unfreeze(base: keras.Model, backbone: str, how="last_block") -> list[str]:
    """Stage 2: unfreeze the last block (or the last N layers). BatchNorm layers stay
    frozen: their running statistics were estimated on ImageNet and updating them with
    small batches damages the pretrained features."""
    base.trainable = True
    names = [l.name for l in base.layers]
    if how == "last_block":
        prefix = BACKBONES[backbone][2]
        start = next(i for i, n in enumerate(names) if n.startswith(prefix))
    else:
        start = max(0, len(names) - int(how))
    for i, l in enumerate(base.layers):
        l.trainable = i >= start and not isinstance(l, layers.BatchNormalization)
    return [l.name for l in base.layers if l.trainable]


def layer_table(model: keras.Model) -> str:
    """Markdown table: layer, output shape, params (for the report, Task 2.2)."""
    rows = ["| layer | type | output shape | params | trainable |", "|---|---|---|---|---|"]
    for l in model.layers:
        shape = getattr(l, "output", None)
        shape = tuple(shape.shape) if shape is not None else "-"
        rows.append(f"| {l.name} | {type(l).__name__} | {shape} | {l.count_params():,} | {l.trainable} |")
    rows.append(f"| **total** | | | {model.count_params():,} | |")
    return "\n".join(rows)
