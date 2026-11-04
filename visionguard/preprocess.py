"""Preprocessing shared by training (src/data.py) and serving (visionguard/inference.py).

Keeping ONE implementation is the whole point: if training resizes with padding and
serving stretches the image, the model sees inputs it was never trained on and its
accuracy silently drops (training/serving skew).

Pipeline for every image, at train time and at serve time:
  1. PIL opens the file and converts it to 8-bit RGB  (palette/RGBA/CMYK/grey -> RGB)
  2. it is re-encoded as JPEG quality 95                (to_rgb_jpeg_bytes)
  3. TensorFlow decodes the JPEG                        (tf.io.decode_jpeg)
  4. resize_with_pad to IMG_SIZE x IMG_SIZE             (keeps aspect ratio, zero bars)
  5. float32 in [0, 255]. Model-specific scaling lives INSIDE the saved model.
"""
from __future__ import annotations

import io
import os

from PIL import Image

IMG_SIZE = int(os.getenv("IMG_SIZE", "224"))
JPEG_QUALITY = 95
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP", "GIF", "TIFF", "MPO"}


def to_rgb_jpeg_bytes(img: Image.Image) -> bytes:
    """Convert any PIL image to RGB and encode it as a JPEG byte string."""
    if img.mode in ("RGBA", "LA", "P"):
        # flatten transparency on white instead of the default black
        img = img.convert("RGBA")
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        img = bg
    else:
        img = img.convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY)
    return buf.getvalue()


def decode_and_resize(jpeg_bytes, img_size: int = IMG_SIZE):
    """TensorFlow part. Works on a scalar string tensor (inside tf.data) or bytes."""
    import tensorflow as tf

    img = tf.io.decode_jpeg(jpeg_bytes, channels=3)
    img = tf.image.resize_with_pad(img, img_size, img_size, antialias=True)
    return tf.cast(img, tf.float32)  # range [0, 255]
