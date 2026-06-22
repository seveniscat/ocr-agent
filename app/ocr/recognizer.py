"""Standalone recognizer helpers (re-recognize a given polygon via PaddleOCR).

Reserved for v2 use cases such as:
- Re-running recognition on a polygon after deskewing it.
- Re-recognizing with a different language model.

v1 keeps detection+recognition fused in :mod:`detector`; this module exists so
the project structure is ready and so tests can target recognition alone.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def crop_polygon(tile: "np.ndarray", polygon: list[list[float]]) -> "np.ndarray":
    """Crop the axis-aligned bbox around a polygon (cheap, no perspective warp).

    Sufficient for line-level text on packaging dielines (mostly axis-aligned).
    Perspective-warp / rotation correction can be added here later.
    """
    import numpy as _np

    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
    h, w = tile.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return _np.zeros((1, 1, 3), dtype="uint8")
    return tile[y1:y2, x1:x2]
