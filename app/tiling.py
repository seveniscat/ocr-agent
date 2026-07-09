"""Tiling engine for very high-resolution images (up to ~10000px).

No single model ingests 10000px natively (DeepSeek OCR / InternVL tile
internally too). Here we:

1. Compute a grid of overlapping tiles from the original image.
2. Yield ``(tile_index, tile_pixels, offset)`` so downstream detectors can run
   per-tile, then add ``offset`` back to map local coords to image coords.
3. Merge duplicate detections that fell in the overlap region using
   PaddleOCR's ``merge_fragmented`` pixel-distance rules (``merge_x_thres`` /
   ``merge_y_thres``), plus containment / substring suppression for seam
   fragments (e.g. a full line in tile 0 + trailing word in tile 1).

Coordinate convention: a *local* point ``(x, y)`` in a tile at ``offset=(dx, dy)``
maps to the global image point ``(x + dx, y + dy)``.
"""
from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image

from .schemas import Item


def image_size(data: bytes) -> tuple[int, int]:
    """Read (width, height) from raw image bytes without full decode."""
    with Image.open(io.BytesIO(data)) as img:
        return img.size  # (width, height)


def load_image(data: bytes) -> np.ndarray:
    """Decode bytes into an RGB numpy array (HxWx3 uint8)."""
    with Image.open(io.BytesIO(data)) as img:
        return np.array(img.convert("RGB"))


# ---------------------------------------------------------------------------
# Grid computation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TileSpec:
    """A planned tile (before pixels are actually cropped)."""

    index: int
    x0: int  # left, inclusive
    y0: int  # top, inclusive
    x1: int  # right, exclusive
    y1: int  # bottom, exclusive


@dataclass(frozen=True)
class GridSpec:
    """The full grid plan for an image."""

    width: int
    height: int
    step: int  # horizontal/vertical step between tile origins
    tile_w: int  # tile width (last column may be clipped)
    tile_h: int  # tile height (last row may be clipped)
    cols: int
    rows: int

    @property
    def count(self) -> int:
        return self.cols * self.rows

    @property
    def needs_tiling(self) -> bool:
        return self.count > 1


def plan_grid(
    width: int,
    height: int,
    target_size: int,
    overlap: float,
) -> GridSpec:
    """Plan a grid for an image.

    - If the image fits within ``target_size`` on both axes → single tile (no overlap).
    - Otherwise compute ``step = round(target_size * (1 - overlap))`` and lay out
      tiles such that the last tile is flush with the image edge (the final gap
      is closed by pulling the last origin left, so no edge pixels are missed).
    """
    if width <= target_size and height <= target_size:
        return GridSpec(
            width=width, height=height, step=0,
            tile_w=width, tile_h=height, cols=1, rows=1,
        )

    step = max(1, round(target_size * (1.0 - overlap)))

    def _origins(length: int) -> list[int]:
        if length <= target_size:
            return [0]
        origins: list[int] = []
        pos = 0
        while pos + target_size < length:
            origins.append(pos)
            pos += step
        # Ensure the final tile is flush with the edge.
        last = length - target_size
        if not origins or origins[-1] != last:
            origins.append(last)
        # Dedup adjacent duplicates that collapse for tiny images.
        return origins

    xs = _origins(width)
    ys = _origins(height)
    return GridSpec(
        width=width, height=height, step=step,
        tile_w=target_size, tile_h=target_size,
        cols=len(xs), rows=len(ys),
    )


def tile_specs(grid: GridSpec) -> list[TileSpec]:
    """Expand a GridSpec into concrete TileSpecs (origin-ordered)."""
    if not grid.needs_tiling:
        return [TileSpec(0, 0, 0, grid.width, grid.height)]

    def _origins(length: int) -> list[int]:
        if length <= grid.tile_w:
            return [0]
        origins: list[int] = []
        pos = 0
        while pos + grid.tile_w < length:
            origins.append(pos)
            pos += grid.step
        last = length - grid.tile_w
        if not origins or origins[-1] != last:
            origins.append(last)
        return origins

    xs = _origins(grid.width)
    ys = _origins(grid.height)
    specs: list[TileSpec] = []
    idx = 0
    for y0 in ys:
        for x0 in xs:
            x1 = min(x0 + grid.tile_w, grid.width)
            y1 = min(y0 + grid.tile_h, grid.height)
            specs.append(TileSpec(idx, x0, y0, x1, y1))
            idx += 1
    return specs


def crop_tile(img: np.ndarray, spec: TileSpec) -> np.ndarray:
    """Crop a tile from an HxWx3 array (slicing is a view, cheap)."""
    return img[spec.y0:spec.y1, spec.x0:spec.x1]


# ---------------------------------------------------------------------------
# Polygon / bbox geometry helpers
# ---------------------------------------------------------------------------


def polygon_to_bbox(poly: list[list[float]]) -> list[float]:
    """Axis-aligned bbox [x1, y1, x2, y2] from a quad polygon."""
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return [min(xs), min(ys), max(xs), max(ys)]


def offset_polygon(
    poly: list[list[float]], dx: float, dy: float
) -> list[list[float]]:
    """Translate a polygon by (dx, dy) — map tile-local → image-global."""
    return [[x + dx, y + dy] for x, y in poly]


def bbox_iou(a: list[float], b: list[float]) -> float:
    """IoU of two axis-aligned boxes [x1,y1,x2,y2]."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (area_a + area_b - inter)


def _bbox_area(bbox: list[float]) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _bbox_extents(bbox: list[float]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = bbox
    return x1, x2, y1, y2


def _bbox_center(bbox: list[float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def _bbox_x_overlap_ratio(a: list[float], b: list[float]) -> float:
    """Overlap of two x-ranges over the smaller width, in [0, 1].

    Like :func:`detector._x_overlap_ratio` but on raw bboxes [x1,y1,x2,y2].
    """
    ax1, _, ax2, _ = a
    bx1, _, bx2, _ = b
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    if inter == 0.0:
        return 0.0
    min_w = min(ax2 - ax1, bx2 - bx1)
    return inter / min_w if min_w > 0 else 0.0


# ---------------------------------------------------------------------------
# Same-line overlap merge (mixed-script detection-split fix)
# ---------------------------------------------------------------------------


def _merge_two_items(a: Item, b: Item) -> Item:
    """Merge two items into one spanning the union of their boxes.

    Geometry: union bbox → axis-aligned quad. Text: concatenated left-to-right
    (by x-center), space-separated; empty/None texts are skipped. The survivor
    keeps the higher confidence and the union polygon. ``recognized`` stays True
    only if both halves were recognized (a merged box containing an
    unrecognized half is marked recognized=False so callers know to re-read).
    """
    ab = a.bbox
    bb = b.bbox
    x1, y1 = min(ab[0], bb[0]), min(ab[1], bb[1])
    x2, y2 = max(ab[2], bb[2]), max(ab[3], bb[3])
    merged_bbox = [x1, y1, x2, y2]
    merged_poly = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

    # Order by x-center for natural left-to-right reading order.
    parts = [(a, _bbox_center(ab)[0]), (b, _bbox_center(bb)[0])]
    parts.sort(key=lambda t: t[1])
    texts = [p.text or p.content for p, _ in parts]
    texts = [t for t in texts if t]
    merged_text = " ".join(texts) if texts else None

    return a.model_copy(
        update={
            "polygon": merged_poly,
            "bbox": merged_bbox,
            "text": merged_text,
            "confidence": max(a.confidence, b.confidence),
            "recognized": bool(a.recognized and b.recognized),
            # drop per-line quads: a merged cross-script box is no longer a
            # clean paragraph block, so `lines` would be misleading.
            "lines": None,
        }
    )


def merge_same_line_overlaps(
    items: list[Item],
    same_line_y_thres: float = 35.0,
    x_overlap_ratio: float = 0.3,
) -> list[Item]:
    """Merge overlapping boxes on the same text line into single boxes.

    Motivation: when one line mixes scripts (e.g. English + Korean) the DB
    detector often splits it into two boxes; after unclipping, those two boxes
    overlap in x. The cross-tile dedupe won't merge them (different text → low
    similarity), so both survive and overlap. This stage merges such pairs into
    one box so every pixel belongs to at most one box.

    Two boxes merge when they are on the same line (top/bottom y-edges within
    ``same_line_y_thres``) AND their x-ranges overlap by ≥ ``x_overlap_ratio``.
    Codes (qr/barcode) and items of different types are left alone.

    Greedy single pass within each row: sorts by x, then folds adjacent
    overlapping boxes left-to-right, so a 3-way overlap collapses to one box.
    """
    if len(items) <= 1:
        return list(items)

    # Only text boxes participate; codes pass through untouched.
    candidates = [it for it in items if it.type == "text"]
    passthrough = [it for it in items if it.type != "text"]
    if len(candidates) <= 1:
        return list(items)

    # Group into "lines" by y proximity (single-link on top edge).
    order = sorted(
        range(len(candidates)),
        key=lambda i: (candidates[i].bbox[1], candidates[i].bbox[0]),
    )
    lines: list[list[int]] = []
    for idx in order:
        it = candidates[idx]
        placed = False
        for line in lines:
            # Compare against the first member's top edge of the line.
            ref_top = candidates[line[0]].bbox[1]
            ref_bot = candidates[line[0]].bbox[3]
            if (
                abs(it.bbox[1] - ref_top) <= same_line_y_thres
                and abs(it.bbox[3] - ref_bot) <= same_line_y_thres
            ):
                line.append(idx)
                placed = True
                break
        if not placed:
            lines.append([idx])

    out: list[Item] = list(passthrough)
    for line_idx in lines:
        # Sort the line's boxes left-to-right by x-center.
        line_idx_sorted = sorted(line_idx, key=lambda i: _bbox_center(candidates[i].bbox)[0])
        current = candidates[line_idx_sorted[0]]
        for j in line_idx_sorted[1:]:
            nxt = candidates[j]
            if _bbox_x_overlap_ratio(current.bbox, nxt.bbox) >= x_overlap_ratio:
                current = _merge_two_items(current, nxt)
            else:
                out.append(current)
                current = nxt
        out.append(current)
    return out


# ---------------------------------------------------------------------------
# Cross-tile deduplication (PaddleOCR merge_fragmented + containment)
# ---------------------------------------------------------------------------


def _text_similarity(a: str | None, b: str | None) -> float:
    """Normalised Levenshtein ratio in [0,1]."""
    a = (a or "").strip()
    b = (b or "").strip()
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    m, n = len(a), len(b)
    if m > n:
        a, b = b, a
        m, n = n, m
    prev = list(range(m + 1))
    for j in range(1, n + 1):
        cur = [j] + [0] * m
        bj = b[j - 1]
        for i in range(1, m + 1):
            cost = 0 if a[i - 1] == bj else 1
            cur[i] = min(cur[i - 1] + 1, prev[i] + 1, prev[i - 1] + cost)
        prev = cur
    dist = prev[m]
    return 1.0 - dist / max(m, n)


def _text_is_substring_part(a: str | None, b: str | None) -> bool:
    """True when the shorter non-empty string appears inside the longer one."""
    a = (a or "").strip()
    b = (b or "").strip()
    if not a or not b:
        return False
    if len(a) <= len(b):
        short, long = a, b
    else:
        short, long = b, a
    return short in long


def _boxes_mergeable_official(
    bbox_a: list[float],
    bbox_b: list[float],
    merge_x_thres: float,
    merge_y_thres: float,
) -> bool:
    """PaddleOCR ``merge_boxes`` rule on axis-aligned bboxes (either order).

    See PaddleOCR ``tools/infer/utility.py``: two boxes on the same text line
  merge when their y-extents are within ``merge_y_thres`` and their x-edges are
    within ``merge_x_thres`` (adjacent seam fragments).
    """
    min_x1, max_x1, min_y1, max_y1 = _bbox_extents(bbox_a)
    min_x2, max_x2, min_y2, max_y2 = _bbox_extents(bbox_b)
    y_ok = (
        abs(min_y1 - min_y2) <= merge_y_thres
        and abs(max_y1 - max_y2) <= merge_y_thres
    )
    if not y_ok:
        return False
    x_gap = min(abs(max_x1 - min_x2), abs(max_x2 - min_x1))
    return x_gap <= merge_x_thres


def _bbox_contained_ratio(inner: list[float], outer: list[float]) -> float:
    """Fraction of ``inner`` area that lies inside ``outer``."""
    ix1, iy1 = max(inner[0], outer[0]), max(inner[1], outer[1])
    ix2, iy2 = min(inner[2], outer[2]), min(inner[3], outer[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    inner_area = _bbox_area(inner)
    if inner_area <= 0.0:
        return 0.0
    return inter / inner_area


def _should_merge_items(
    a: Item,
    b: Item,
    merge_x_thres: float,
    merge_y_thres: float,
    text_threshold: float,
) -> bool:
    """True when ``b`` is a duplicate / fragment of ``a`` and should be dropped."""
    if a.type != b.type:
        return False

    ta = a.text or a.content
    tb = b.text or b.content
    no_text = not ta and not tb

    # 1) Official seam merge (geometry only — codes or identical OCR noise).
    if _boxes_mergeable_official(a.bbox, b.bbox, merge_x_thres, merge_y_thres):
        if no_text:
            return True
        if ta and tb and _text_similarity(ta, tb) >= text_threshold:
            return True

    # 2) Containment: smaller box inside larger on the same line + text substring.
    area_a, area_b = _bbox_area(a.bbox), _bbox_area(b.bbox)
    if area_a > 0 and area_b > 0:
        if area_a >= area_b:
            outer, inner, outer_t, inner_t = a, b, ta, tb
        else:
            outer, inner, outer_t, inner_t = b, a, tb, ta
        contain = _bbox_contained_ratio(inner.bbox, outer.bbox)
        min_xo, max_xo, min_yo, max_yo = _bbox_extents(outer.bbox)
        min_xi, max_xi, min_yi, max_yi = _bbox_extents(inner.bbox)
        y_aligned = (
            abs(min_yo - min_yi) <= merge_y_thres
            and abs(max_yo - max_yi) <= merge_y_thres
        )
        x_inside = (
            min_xi >= min_xo - merge_x_thres
            and max_xi <= max_xo + merge_x_thres
        )
        if y_aligned and (contain >= 0.75 or x_inside):
            if no_text:
                return True
            if outer_t and inner_t and _text_is_substring_part(inner_t, outer_t):
                return True

    # 3) High-IoU near-duplicate (legacy path for same-text tile overlap).
    if bbox_iou(a.bbox, b.bbox) >= 0.5:
        if no_text:
            return True
        if ta and tb and _text_similarity(ta, tb) >= text_threshold:
            return True

    return False


def dedupe_items(
    items: list[Item],
    merge_x_thres: float = 50.0,
    merge_y_thres: float = 35.0,
    text_threshold: float = 0.6,
) -> list[Item]:
    """Merge cross-tile duplicates using PaddleOCR slice merge thresholds.

    Survivor priority for text items: longer text (substring cases) → larger
    bbox → higher confidence. This keeps the full-line detection when a tile
    seam also produced a trailing fragment (``completa.`` inside a full sentence).
    """
    if len(items) <= 1:
        return list(items)

    def _sort_key(it: Item) -> tuple:
        text = (it.text or it.content or "")
        return (len(text), _bbox_area(it.bbox), it.confidence)

    order = sorted(range(len(items)), key=lambda i: _sort_key(items[i]), reverse=True)
    suppressed = [False] * len(items)
    out: list[Item] = []

    for i_pos, i in enumerate(order):
        if suppressed[i_pos]:
            continue
        item_i = items[i]
        out.append(item_i)
        for j_pos in range(i_pos + 1, len(order)):
            if suppressed[j_pos]:
                continue
            j = order[j_pos]
            item_j = items[j]
            if _should_merge_items(
                item_i,
                item_j,
                merge_x_thres,
                merge_y_thres,
                text_threshold,
            ):
                suppressed[j_pos] = True
    return out


def renumber(items: list[Item], prefix: str = "t") -> list[Item]:
    """Re-assign sequential ids ``{prefix}_001`` after dedup."""
    return [
        item.model_copy(update={"id": f"{prefix}_{i + 1:03d}"})
        for i, item in enumerate(items)
    ]
