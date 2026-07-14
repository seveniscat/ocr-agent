"""Detect circular / ring-shaped text regions on packaging images.

Circular text (around logos, seals, badges, caps) is a hard case for line-based
OCR: the characters sit on an arc, so the recognizer's left-to-right reading
assumption breaks. This module LOCALIZES such regions with pure geometry so the
pipeline can hand the whole ring to the VLM with a specialized prompt — we do
not try to recognize the text here, only find the ring.

Algorithm (no model, no VLM call — pure OpenCV + numpy):
  1. ``cv2.HoughCircles`` on a grayscale + blurred + thresholded image to find
     circle candidates. Radius bounds are derived from the image size (a real
     circular logo is typically 5%–40% of the shorter edge).
  2. For each candidate circle (cx, cy, r), collect the text items whose quad
     CENTERS fall inside the annular band ``|dist(center, (cx,cy)) - r| ≤
     band_ratio * r`` — i.e. the characters sit ON the circle, not inside it.
  3. Validate it's truly arc-arranged, not a coincidence: the per-item baseline
     angles (quad edge p0→p1) must vary across the ring. If every member has
     nearly the same angle it's a straight line that happened to fall near a
     found circle, so we reject it via an angle-spread threshold.
  4. Require at least ``min_members`` items in the band — a single arc fragment
     is not enough to claim a ring.

Everything is wrapped so a missing cv2 / a degenerate image returns an empty
list: circular detection is a GAIN-only channel. It must never break the main
OCR flow on images that have no circular text (the common case).
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np

    from .schemas import Item

logger = logging.getLogger(__name__)


@dataclass
class CircularRegion:
    """One ring-shaped text region found on the image.

    ``bbox`` is the axis-aligned bounding box of the circle's outer extent
    (center ± radius), in image pixels — this is what gets cropped and sent to
    the VLM. ``member_indices`` are the positions of the constituent text items
    in the original list, so the caller can (a) avoid re-sending them as plain
    low-confidence suspects and (b) write the VLM result onto a representative
    member. ``polygon`` is the bbox as a 4-point quad for API compatibility.
    """

    cx: float
    cy: float
    radius: float
    bbox: list[float]                       # [x1, y1, x2, y2]
    polygon: list[list[float]]              # 4-point quad of the bbox
    member_indices: list[int] = field(default_factory=list)


def detect_circular_regions(
    img: "np.ndarray",
    items: "list[Item]",
    settings,
) -> list[CircularRegion]:
    """Find circular text regions on the image. Never raises.

    ``items`` are the OCR text items (only their ``polygon`` is read). Returns
    a list of :class:`CircularRegion`, possibly empty. Safe to call on images
    with no circular text — returns ``[]``.

    The whole body is guarded: if OpenCV is unavailable, the image is too small
    for HoughCircles, or anything unexpected happens, we log + return ``[]``.
    This keeps circular detection as an opt-in gain that can never destabilize
    the main OCR path.
    """
    if not getattr(settings, "circular_detect_enabled", True):
        return []
    # Only text items carry the quad geometry we analyze. Codes (qr/barcode) and
    # already-fallback items are irrelevant to circular detection.
    text_items = [
        (i, it) for i, it in enumerate(items)
        if it.type == "text" and it.polygon and len(it.polygon) >= 4
    ]
    if len(text_items) < getattr(settings, "circular_min_members", 4):
        return []

    try:
        import cv2
        import numpy as np
    except ImportError:  # pragma: no cover — opencv is a hard dep, but be safe
        logger.warning("regions: opencv unavailable; circular detection skipped")
        return []

    h, w = img.shape[:2]
    circles = _find_circles(cv2, np, img, settings)
    if not circles:
        return []

    band_ratio = getattr(settings, "circular_band_ratio", 0.25)
    min_members = getattr(settings, "circular_min_members", 4)
    regions: list[CircularRegion] = []
    claimed: set[int] = set()  # an item belongs to at most one ring

    for cx, cy, r in circles:
        members = _collect_band_members(
            text_items, cx, cy, r, band_ratio, excluded=claimed
        )
        if len(members) < min_members:
            continue
        # Angle-spread check: reject a straight line that merely passes near a
        # found circle. On a real ring the per-item baseline angles span a wide
        # range (they point in different directions around the circle).
        if not _is_arc_arranged(members, cx, cy):
            continue

        idxs = [i for i, _ in members]
        claimed.update(idxs)
        x1, y1 = cx - r, cy - r
        x2, y2 = cx + r, cy + r
        regions.append(
            CircularRegion(
                cx=cx, cy=cy, radius=r,
                bbox=[x1, y1, x2, y2],
                polygon=[[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
                member_indices=idxs,
            )
        )

    if regions:
        logger.info(
            "regions: %d circular region(s) found (%d member items total)",
            len(regions), sum(len(r.member_indices) for r in regions),
        )
    return regions


# ---------------------------------------------------------------------------
# Circle finding
# ---------------------------------------------------------------------------


def _find_circles(cv2, np, img, settings):
    """Run HoughCircles with radius bounds derived from the image size.

    Returns a list of ``(cx, cy, r)`` tuples (floats). Empty on any failure —
    the caller treats that as "no circles", not an error.
    """
    h, w = img.shape[:2]
    short_edge = min(h, w)
    if short_edge < 64:
        return []
    # A real circular logo/seal on packaging is typically 5%–40% of the shorter
    # edge. HoughCircles wants minRadius < maxRadius and both > 0.
    min_r = max(8, int(short_edge * 0.05))
    max_r = int(short_edge * 0.40)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    blur = cv2.medianBlur(gray, 5)
    try:
        found = cv2.HoughCircles(
            blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=min_r * 2,
            param1=100, param2=30,
            minRadius=min_r, maxRadius=max_r,
        )
    except cv2.error:  # pragma: no cover — degenerate image
        return []
    if found is None:
        return []
    # found shape: (1, N, 3) as float32. Yield (cx, cy, r) tuples.
    return [(float(c[0]), float(c[1]), float(c[2])) for c in found[0]]


# ---------------------------------------------------------------------------
# Geometry: band membership + arc validation
# ---------------------------------------------------------------------------


def _collect_band_members(
    text_items, cx, cy, r, band_ratio, excluded
):
    """Return the ``(index, item)`` pairs whose quad center sits in the annulus.

    The annulus is centered on radius ``r`` with half-width ``band_ratio * r``
    — characters ON the circle, not inside the disk. Items already claimed by
    another ring are skipped.
    """
    half_band = band_ratio * r
    members = []
    for i, it in text_items:
        if i in excluded:
            continue
        mx, my = _quad_center(it.polygon)
        d = math.hypot(mx - cx, my - cy)
        if abs(d - r) <= half_band:
            members.append((i, it))
    return members


def _is_arc_arranged(members, cx, cy) -> bool:
    """True when the member angles span a wide range (i.e. really around a ring).

    Computes each member's azimuth from the circle center, then requires the
    angular span (max - min, circularly) to exceed 60°. A straight line of text
    passing near the circle would have all members at nearly the same azimuth
    and be rejected here.
    """
    if len(members) < 2:
        return False
    angles = []
    for _, it in members:
        mx, my = _quad_center(it.polygon)
        angles.append(math.atan2(my - cy, mx - cx))  # radians, [-π, π]
    # Circular spread: convert to a 0..2π sorted list and find the largest gap;
    # span = 2π - largest_gap. This handles wrap-around at ±π correctly.
    sorted_a = sorted(angles)
    max_gap = 0.0
    for i in range(len(sorted_a)):
        a0 = sorted_a[i]
        a1 = sorted_a[(i + 1) % len(sorted_a)]
        gap = (a1 - a0) % (2 * math.pi)
        if gap > max_gap:
            max_gap = gap
    span = 2 * math.pi - max_gap
    return span >= math.radians(60)


def _quad_center(polygon) -> tuple[float, float]:
    """Centroid of a quad (average of its points)."""
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return sum(xs) / len(xs), sum(ys) / len(ys)
