"""Tests for the interactive panel-splitting helpers.

Pure-function tests for ``compute_panels`` (no image, no cv2) and a couple of
synthetic-image checks for ``detect_candidate_lines``. Mirrors the style of
``test_paragraph.py`` / ``test_preprocess.py``.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from app.panels import CONF_HIGH, CONF_MID, CONF_LOW, compute_panels, detect_candidate_lines


# ---------------------------------------------------------------------------
# compute_panels — pure geometry
# ---------------------------------------------------------------------------


def test_no_lines_returns_whole_image():
    """Zero confirmed cut lines → the whole image is one panel."""
    out = compute_panels([], [], width=400, height=300)
    assert len(out) == 1
    assert out[0].bbox == [0, 0, 400, 300]
    assert out[0].width == 400 and out[0].height == 300


def test_one_h_one_v_makes_four_panels():
    """1 horizontal + 1 vertical line splits into a 2×2 grid."""
    out = compute_panels([150], [200], width=400, height=300)
    assert len(out) == 4
    # Panels are sorted top→bottom, left→right.
    assert out[0].bbox == [0, 0, 200, 150]      # top-left
    assert out[1].bbox == [200, 0, 400, 150]    # top-right
    assert out[2].bbox == [0, 150, 200, 300]    # bottom-left
    assert out[3].bbox == [200, 150, 400, 300]  # bottom-right


def test_only_horizontal_lines():
    """3 horizontal lines, no vertical → 4 horizontal strips."""
    out = compute_panels([100, 200, 300], [], width=400, height=400)
    assert len(out) == 4
    assert out[0].bbox == [0, 0, 400, 100]
    assert out[3].bbox == [0, 300, 400, 400]


def test_unsorted_and_duplicate_inputs_are_cleaned():
    """Caller may pass unsorted / near-duplicate / out-of-range positions."""
    out = compute_panels(
        h_positions=[300, 100, 100, 101, -5, 9999],  # dup + out-of-range
        v_positions=[200, 201, 50],                  # near-dup
        width=400, height=400,
    )
    # After cleaning: h=[100,300], v=[50,200] → 3 rows × 3 cols = 9 panels.
    assert len(out) == 9


def test_line_on_edge_is_dropped():
    """A line exactly on the image edge adds no split (just the boundary)."""
    out = compute_panels([0, 300, 400], [0, 400], width=400, height=400)
    # Only the interior line [300] (h) counts; v has none → 2 rows × 1 col.
    assert len(out) == 2
    assert out[0].bbox == [0, 0, 400, 300]
    assert out[1].bbox == [0, 300, 400, 400]


def test_panel_dimensions_match_bbox():
    out = compute_panels([100], [150], width=300, height=200)
    for p in out:
        assert p.width == p.bbox[2] - p.bbox[0]
        assert p.height == p.bbox[3] - p.bbox[1]


# ---------------------------------------------------------------------------
# detect_candidate_lines — synthetic images
# ---------------------------------------------------------------------------


def _white_with_lines(w, h, h_ys=(), v_xs=(), line_w=3):
    """A white image with black horizontal (at y=h_ys) and vertical (at x=v_xs) lines."""
    arr = np.full((h, w, 3), 255, dtype=np.uint8)
    for y in h_ys:
        arr[max(0, y - line_w) : y + line_w, :] = 0
    for x in v_xs:
        arr[:, max(0, x - line_w) : x + line_w] = 0
    return Image.fromarray(arr)


def test_detects_strong_horizontal_line():
    """A single full-width black line should be detected as a high-conf candidate."""
    img = _white_with_lines(400, 300, h_ys=[150])
    res = detect_candidate_lines(img)
    # At least one horizontal candidate near y=150.
    near = [ln for ln in res["h_lines"] if abs(ln["pos"] - 150) < 15]
    assert near, f"expected a candidate near y=150, got {res['h_lines']}"
    assert near[0]["confidence"] == CONF_HIGH
    assert near[0]["selected"] is True


def test_detects_strong_vertical_line():
    img = _white_with_lines(400, 300, v_xs=[200])
    res = detect_candidate_lines(img)
    near = [ln for ln in res["v_lines"] if abs(ln["pos"] - 200) < 15]
    assert near, f"expected a candidate near x=200, got {res['v_lines']}"
    assert near[0]["confidence"] == CONF_HIGH


def test_returns_original_image_dimensions():
    img = _white_with_lines(567, 489)
    res = detect_candidate_lines(img)
    assert res["width"] == 567
    assert res["height"] == 489


def test_blank_image_has_no_high_conf_lines():
    """A pure-white image should not produce high-confidence candidates."""
    img = _white_with_lines(400, 300)
    res = detect_candidate_lines(img)
    hi = [ln for ln in (res["h_lines"] + res["v_lines"]) if ln["confidence"] == CONF_HIGH]
    assert hi == []
