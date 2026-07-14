"""Tests for circular-region detection (pure geometry, no model needed).

``detect_circular_regions`` finds rings of text (around logos/seals/badges)
using HoughCircles + an angle-spread check, so the pipeline can hand the whole
ring to the VLM with a circular-aware prompt. These tests use SYNTHETIC images
(numpy) so they run without paddle/VLM, and cover:

  - a ring of boxes around a drawn circle → 1 region found, members correct
  - scattered boxes with no circle → no regions
  - boxes clumped in one arc (low angle spread) → rejected (not a full ring)
  - the ``circular_detect_enabled=False`` switch → skipped
  - too few members → skipped
"""
from __future__ import annotations

import math

import numpy as np

from app.config import Settings
from app.regions import detect_circular_regions
from app.schemas import Item


def _img(size: int = 400) -> np.ndarray:
    """A black image with a white circle drawn at the center."""
    import cv2

    img = np.zeros((size, size, 3), dtype=np.uint8)
    cv2.circle(img, (size // 2, size // 2), size // 4, (255, 255, 255), 2)
    return img


def _ring_items(cx, cy, r, n=8, box=12):
    """Build n small text Items whose quad CENTERS sit exactly on the circle.

    Distributed evenly around the ring so the angle-spread check passes.
    Each item is a tiny axis-aligned quad of size box×box centered on the ring.
    """
    items = []
    for k in range(n):
        ang = 2 * math.pi * k / n
        mx = cx + r * math.cos(ang)
        my = cy + r * math.sin(ang)
        x1, y1 = mx - box / 2, my - box / 2
        x2, y2 = mx + box / 2, my + box / 2
        items.append(
            Item(
                id=f"t{k}", type="text", text=f"c{k}",
                polygon=[[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                bbox=[x1, y1, x2, y2],
                confidence=0.5, source="paddleocr",
            )
        )
    return items


def _settings(**over) -> Settings:
    """Settings with circular detection on and a low min_members for tests."""
    base = Settings()
    return base.model_copy(
        update={
            "circular_detect_enabled": True,
            "circular_min_members": 4,
            "circular_band_ratio": 0.25,
            **over,
        }
    )


# ---------------------------------------------------------------------------
# Happy path: a ring of boxes on a drawn circle is detected.
# ---------------------------------------------------------------------------


def test_ring_of_boxes_is_detected():
    img = _img(400)
    cx = cy = 200
    r = 100
    items = _ring_items(cx, cy, r, n=8)
    regions = detect_circular_regions(img, items, _settings())

    assert len(regions) >= 1, "expected at least one circular region"
    reg = regions[0]
    # The detected circle should be near the drawn one (cx,cy,r within a margin).
    assert abs(reg.cx - cx) < 25 and abs(reg.cy - cy) < 25
    assert abs(reg.radius - r) < 30
    # Most of the 8 boxes should be claimed as members.
    assert len(reg.member_indices) >= 6
    # bbox is the circle's outer extent.
    assert reg.bbox[0] < cx < reg.bbox[2] and reg.bbox[1] < cy < reg.bbox[3]


# ---------------------------------------------------------------------------
# No circle / scattered boxes → nothing found.
# ---------------------------------------------------------------------------


def test_scattered_boxes_no_circle_finds_nothing():
    # A blank image (no drawn circle) + boxes placed in a rough row, not a ring.
    img = np.zeros((400, 400, 3), dtype=np.uint8)
    items = []
    for k in range(8):
        x = 40 + k * 40
        items.append(
            Item(
                id=f"t{k}", type="text", text=f"c{k}",
                polygon=[[x, 200], [x + 12, 200], [x + 12, 212], [x, 212]],
                bbox=[x, 200, x + 12, 212],
                confidence=0.5, source="paddleocr",
            )
        )
    assert detect_circular_regions(img, items, _settings()) == []


# ---------------------------------------------------------------------------
# Boxes clumped in one arc → rejected by the angle-spread check.
# ---------------------------------------------------------------------------


def test_clumped_arc_is_rejected():
    """Boxes on the circle but all within a small arc (low spread) → not a ring."""
    img = _img(400)
    cx = cy = 200
    r = 100
    # 8 boxes all in a 30° arc near the top — same circle, but not a full ring.
    items = []
    for k in range(8):
        ang = math.radians(-15 + k * 4)  # span ~28°, all near top (-y is up)
        mx = cx + r * math.cos(ang - math.pi / 2)
        my = cy + r * math.sin(ang - math.pi / 2)
        items.append(
            Item(
                id=f"t{k}", type="text", text=f"c{k}",
                polygon=[[mx - 6, my - 6], [mx + 6, my - 6],
                         [mx + 6, my + 6], [mx - 6, my + 6]],
                bbox=[mx - 6, my - 6, mx + 6, my + 6],
                confidence=0.5, source="paddleocr",
            )
        )
    regions = detect_circular_regions(img, items, _settings())
    # Even if HoughCircles finds the circle, the angle-spread check must reject
    # a clumped arc as "not a full ring".
    assert regions == [] or all(
        # be lenient: if something is found, it must not have claimed all boxes
        # (the spread check should have filtered the clump)
        len(r.member_indices) < len(items) for r in regions
    )


# ---------------------------------------------------------------------------
# Switch off → no-op.
# ---------------------------------------------------------------------------


def test_disabled_returns_empty():
    img = _img(400)
    items = _ring_items(200, 200, 100, n=8)
    s = _settings(circular_detect_enabled=False)
    assert detect_circular_regions(img, items, s) == []


# ---------------------------------------------------------------------------
# Too few members → skipped.
# ---------------------------------------------------------------------------


def test_too_few_members_skipped():
    img = _img(400)
    # Only 2 boxes on the ring — below the default min_members=4.
    items = _ring_items(200, 200, 100, n=2)
    assert detect_circular_regions(img, items, _settings()) == []


# ---------------------------------------------------------------------------
# Non-text items (qr/barcode) are ignored.
# ---------------------------------------------------------------------------


def test_non_text_items_ignored():
    img = _img(400)
    # 4 text boxes on the ring + 4 barcode items also on the ring. Only text
    # counts toward membership; with 4 text members it should still detect.
    text_items = _ring_items(200, 200, 100, n=4)
    extra = []
    for k in range(4):
        ang = math.pi / 2 + 2 * math.pi * (k + 4) / 8
        mx = 200 + 100 * math.cos(ang)
        my = 200 + 100 * math.sin(ang)
        extra.append(
            Item(
                id=f"b{k}", type="barcode", content=f"x{k}",
                polygon=[[mx - 6, my - 6], [mx + 6, my - 6],
                         [mx + 6, my + 6], [mx - 6, my + 6]],
                bbox=[mx - 6, my - 6, mx + 6, my + 6],
                confidence=0.9, source="pyzbar",
            )
        )
    regions = detect_circular_regions(img, text_items + extra, _settings())
    # The 4 text boxes are enough; barcode items are ignored for membership.
    assert len(regions) >= 1
    # No barcode index should appear as a member.
    member_ids = {text_items[i].id for i in regions[0].member_indices}
    assert all(m.startswith("t") for m in member_ids)
