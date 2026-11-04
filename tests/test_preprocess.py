"""Training and serving must produce the same tensor for the same image."""
import io

import numpy as np
import pytest
from PIL import Image

tf = pytest.importorskip("tensorflow")

from visionguard.preprocess import decode_and_resize, to_rgb_jpeg_bytes  # noqa: E402


def test_shape_and_range():
    img = Image.new("RGBA", (300, 120), (10, 200, 30, 128))
    x = decode_and_resize(tf.constant(to_rgb_jpeg_bytes(img)), 224).numpy()
    assert x.shape == (224, 224, 3) and x.min() >= 0 and x.max() <= 255
    assert np.allclose(x[0, 112], 0) and np.allclose(x[-1, 112], 0)   # padding bars, aspect kept


def test_train_equals_serve(tmp_path):
    rng = np.random.default_rng(0)
    img = Image.fromarray(rng.integers(0, 255, (90, 140, 3), dtype=np.uint8))
    p = tmp_path / "a.jpg"
    p.write_bytes(to_rgb_jpeg_bytes(img))                       # what src.prepare writes
    train = decode_and_resize(tf.io.read_file(str(p)), 128)     # what src.data reads
    buf = io.BytesIO()
    img.save(buf, "PNG")
    serve_img = Image.open(io.BytesIO(buf.getvalue()))
    serve = decode_and_resize(tf.constant(to_rgb_jpeg_bytes(serve_img)), 128)  # what inference does
    assert np.array_equal(train.numpy(), serve.numpy())
