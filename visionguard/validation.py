"""Upload validation shared by the web app and the model API (Task 4.2).

Three independent checks, because each catches a different attack or mistake:
  1. extension allow-list      -> wrong file types (.pdf, .exe, .txt ...)
  2. byte size limit           -> huge uploads (also enforced earlier by MAX_CONTENT_LENGTH)
  3. real decode with Pillow   -> files renamed to .jpg that are not images, truncated
                                  files, and decompression bombs (tiny file, huge pixels)
The image is only ever held in memory (bytes); nothing is written to disk.
"""
from __future__ import annotations

import os

from PIL import Image
from werkzeug.utils import secure_filename

from visionguard.inference import load_image_bytes
from visionguard.preprocess import ALLOWED_FORMATS

ALLOWED_EXT = {"jpg", "jpeg", "png", "webp", "bmp", "gif"}
MAX_UPLOAD_MB = float(os.getenv("MAX_UPLOAD_MB", "8"))
Image.MAX_IMAGE_PIXELS = 40_000_000  # ~40 MP; larger raises DecompressionBombError


class UploadError(Exception):
    def __init__(self, status: int, title: str, message: str):
        super().__init__(message)
        self.status, self.title, self.message = status, title, message


def validate_upload(filename: str | None, raw: bytes, max_mb: float = MAX_UPLOAD_MB):
    """Returns (safe_filename, PIL.Image). Raises UploadError with an HTTP status."""
    if not filename or not raw:
        raise UploadError(400, "No file", "Please choose an image to upload.")
    safe = secure_filename(filename) or "upload"
    ext = safe.rsplit(".", 1)[-1].lower() if "." in safe else ""
    if ext not in ALLOWED_EXT:
        raise UploadError(415, "Unsupported file type",
                          f"'.{ext or '?'}' is not allowed. Upload one of: {', '.join(sorted(ALLOWED_EXT))}.")
    if len(raw) > max_mb * 1024 * 1024:
        raise UploadError(413, "File too large", f"The limit is {max_mb:g} MB.")
    try:
        img = load_image_bytes(raw)
    except Image.DecompressionBombError:
        raise UploadError(422, "Image too large", "The image has too many pixels to process safely.")
    except Exception:  # noqa: BLE001
        raise UploadError(422, "Not a readable image",
                          "The file has an image extension but could not be decoded. It may be corrupt "
                          "or not an image at all.")
    if img.format not in ALLOWED_FORMATS:
        raise UploadError(415, "Unsupported image format", f"Format {img.format} is not supported.")
    return safe, img
