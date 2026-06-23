"""Tests for die-line auto-crop preprocessing.

Pure algorithmic tests; no paddle / pyzbar / network needed — only numpy + cv2,
both already required by the OCR path. Mirrors the style of test_paragraph.py.
"""
from __future__ import annotations

import numpy as np

from app.preprocess import autocrop, content_bbox


def _white(h: int, w: int, rgb=(255, 255, 255)) -> np.ndarray:
    """An all-white HxWx3 uint8 image."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:] = rgb
    return img


def _draw_rect(img: np.ndarray, x0: int, y0: int, x1: int, y1: int, color=(0, 0, 0)):
    """Draw a filled rectangle (axis-aligned) — simulates die-line ink."""
    img[y0:y1, x0:x1] = color


# ---------------------------------------------------------------------------
# content_bbox
# ---------------------------------------------------------------------------


def test_content_bbox_all_white_returns_none():
    assert content_bbox(_white(100, 80)) is None


def test_content_bbox_finds_ink_bbox():
    # White 200x150 image with a black 10x10 block at (x0,y0)=(20,30).
    img = _white(150, 200)
    _draw_rect(img, 20, 30, 30, 40)
    # bbox is slice-friendly (exclusive end) → (20, 30, 30, 40).
    assert content_bbox(img) == (20, 30, 30, 40)


def test_content_bbox_takes_union_of_disjoint_ink():
    # Two ink regions: bottom-right of one and top-left of other define the union.
    img = _white(200, 200)
    _draw_rect(img, 10, 10, 20, 20)        # top-left
    _draw_rect(img, 150, 170, 180, 190)    # bottom-right
    assert content_bbox(img) == (10, 10, 180, 190)


def test_content_bbox_threshold_filters_faint_ink():
    # Light-grey (200) ink: at threshold=240 it counts as ink; at threshold=150 it doesn't.
    img = _white(100, 100)
    _draw_rect(img, 20, 20, 60, 60, color=(200, 200, 200))
    assert content_bbox(img, threshold=240) == (20, 20, 60, 60)
    assert content_bbox(img, threshold=150) is None


def test_content_bbox_accepts_grayscale_input():
    # content_bbox must also work on a 2D (single-channel) array.
    gray = np.full((80, 120), 255, dtype=np.uint8)
    gray[10:30, 40:70] = 0
    assert content_bbox(gray) == (40, 10, 70, 30)


# ---------------------------------------------------------------------------
# autocrop
# ---------------------------------------------------------------------------


def test_autocrop_blank_returns_original_and_none():
    img = _white(60, 40)
    out, crop = autocrop(img)
    assert crop is None
    assert out.shape == img.shape
    assert np.array_equal(out, img)


def test_autocrop_trims_to_content():
    img = _white(150, 200)
    _draw_rect(img, 20, 30, 30, 40)
    out, crop = autocrop(img)
    assert crop == [20, 30, 30, 40]
    assert out.shape == (10, 10, 3)


def test_autocrop_padding_expands_box():
    img = _white(150, 200)
    _draw_rect(img, 50, 50, 60, 60)
    out, crop = autocrop(img, padding=5)
    assert crop == [45, 45, 65, 65]
    assert out.shape == (20, 20, 3)


def test_autocrop_padding_clamps_to_image_edge():
    # Ink touches the top-left corner; padding must not push the box negative.
    img = _white(100, 100)
    _draw_rect(img, 0, 0, 10, 10)
    _, crop = autocrop(img, padding=5)
    assert crop == [0, 0, 15, 15]


def test_autocrop_returns_independent_copy():
    # Mutating the output must not affect the input image.
    img = _white(50, 50)
    _draw_rect(img, 10, 10, 20, 20)
    out, _ = autocrop(img)
    out[:] = 123
    # Original still has its ink rectangle at [10:20,10:20].
    assert (img[10:20, 10:20] == 0).all()
    assert (img[0:10, :] == 255).all()
