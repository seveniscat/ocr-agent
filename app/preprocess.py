"""Image preprocessing: auto-crop the blank margins of a packaging dieline.

Packaging die-lines are typically drawn on a near-white background with the
artwork (cut/crease lines, text, tick marks) clustered in a central region
surrounded by wide margins — scan whitespace, trim waste, registration gutters.
Cropping those margins before tiling means:

  - fewer tiles → less OCR work and fewer cross-seam merges;
  - the detector spends its resolution budget on real content;
  - all downstream coordinates live in the cropped space (zero remap), and the
    one ``crop`` offset in the response lets callers map back to the original.

The "subject" here is intentionally defined as the **content bounding box** —
the tightest rectangle enclosing every non-background pixel — which is the most
robust definition for standard die-lines (multiple disconnected cut lines,
asymmetric layouts, etc.). Detecting the single largest closed contour is more
semantic but brittle when several cut lines compete for "main outline".

No new dependency: OpenCV + NumPy are already required by the OCR path.
"""
from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


def content_bbox(
    img: np.ndarray, *, threshold: int = 240
) -> Optional[tuple[int, int, int, int]]:
    """Tight bounding box of all "ink" pixels, or ``None`` if the image is blank.

    "Ink" = any pixel whose grayscale value is strictly below ``threshold``
    (default 240 catches near-white backgrounds while tolerating off-white
    paper / anti-aliasing). Returns ``(x0, y0, x1, y1)`` in **exclusive** pixel
    coords (i.e. directly usable as ``img[y0:y1, x0:x1]``).

    Returns ``None`` for a fully-blank image so the caller can fall back to the
    original without special-casing.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    # Boolean mask of foreground (ink) pixels.
    mask = gray < threshold
    if not mask.any():
        return None
    ys, xs = np.where(mask)
    # +1 → exclusive upper bound, so the bbox is slice-friendly.
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def autocrop(
    img: np.ndarray,
    *,
    threshold: int = 240,
    padding: int = 0,
) -> tuple[np.ndarray, Optional[list[int]]]:
    """Crop blank margins off ``img``; return ``(cropped, crop_box)``.

    ``crop_box`` is ``[x0, y0, x1, y1]`` in the **original** image's coords
    (exclusive end), or ``None`` when nothing was cropped (blank image). The
    returned array is a copy, so downstream in-place edits never alias the input.
    ``padding`` keeps that many extra pixels of margin on every side (clamped to
    the image bounds).
    """
    h, w = img.shape[:2]
    bbox = content_bbox(img, threshold=threshold)
    if bbox is None:
        return img, None

    x0, y0, x1, y1 = bbox
    pad = max(0, int(padding))
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(w, x1 + pad), min(h, y1 + pad)

    # Degenerate (e.g. padding collapsed a 1px box against an edge): skip.
    if x1 <= x0 or y1 <= y0:
        return img, None

    cropped = img[y0:y1, x0:x1].copy()
    return cropped, [x0, y0, x1, y1]
