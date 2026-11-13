"""Unicode-safe image I/O.

``cv2.imread`` / ``cv2.imwrite`` cannot open paths that contain non-ASCII
characters on Windows: they return ``None`` (read) or fail silently (write).
That matters here because the dataset is often unpacked under a path such as
``data/raw/LoveDA/...``. These wrappers read the raw bytes with numpy and let
OpenCV decode them, which works for any path.

Use them exactly like the OpenCV originals::

    from imgio import imread, imwrite
    img = imread(path, cv2.IMREAD_GRAYSCALE)
    imwrite(out_path, img)
"""
import os

import cv2
import numpy as np


def imread(path, flags=cv2.IMREAD_COLOR):
    """Read an image from ``path``; returns None if the file is empty/unreadable."""
    try:
        buf = np.fromfile(os.fspath(path), dtype=np.uint8)
    except (OSError, ValueError):
        return None
    if buf.size == 0:
        return None
    return cv2.imdecode(buf, flags)


def imwrite(path, img, params=None):
    """Write ``img`` to ``path``; the extension decides the encoder."""
    path = os.fspath(path)
    ext = os.path.splitext(path)[1] or ".png"
    ok, buf = cv2.imencode(ext, img, params or [])
    if not ok:
        return False
    buf.tofile(path)
    return True
