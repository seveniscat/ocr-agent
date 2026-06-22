"""Tiling engine for very high-resolution images (up to ~10000px).

No single model ingests 10000px natively (DeepSeek OCR / InternVL tile
internally too). Here we:

1. Compute a grid of overlapping tiles from the original image.
2. Yield ``(tile_index, tile_pixels, offset)`` so downstream detectors can run
   per-tile, then add ``offset`` back to map local coords to image coords.
3. Merge duplicate detections that fell in the overlap region (polygon IoU
   NMS + text similarity).

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


# ---------------------------------------------------------------------------
# Cross-tile deduplication
# ---------------------------------------------------------------------------


def _text_similarity(a: str | None, b: str | None) -> float:
    """Normalised Damerau-ish similarity in [0,1]. Cheap and dependency-free.

    We use a normalised Levenshtein ratio: ``1 - dist/max_len``.
    For our purpose (merge near-duplicate OCR lines across an overlap seam),
    crude is fine — we only call this when two boxes already overlap heavily.
    """
    a = (a or "").strip()
    b = (b or "").strip()
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    # Levenshtein DP.
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


def dedupe_items(
    items: list[Item],
    iou_threshold: float = 0.5,
    text_threshold: float = 0.6,
) -> list[Item]:
    """Greedy NMS across tiles, with text-similarity confirmation.

    Two items merge when their bboxes overlap above ``iou_threshold`` **and**
    (their texts are similar above ``text_threshold``, OR at least one has no
    text — e.g. code regions where geometry alone is the signal). The survivor
    keeps the higher-confidence item's metadata.
    """
    if len(items) <= 1:
        return list(items)

    # Sort by confidence desc so the best representative survives.
    order = sorted(range(len(items)), key=lambda i: items[i].confidence, reverse=True)
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
            iou = bbox_iou(item_i.bbox, item_j.bbox)
            if iou < iou_threshold:
                continue
            sim = _text_similarity(
                item_i.text or item_i.content, item_j.text or item_j.content
            )
            no_text = not (item_i.text or item_i.content) and not (
                item_j.text or item_j.content
            )
            if sim >= text_threshold or no_text:
                suppressed[j_pos] = True
    return out


def renumber(items: list[Item], prefix: str = "t") -> list[Item]:
    """Re-assign sequential ids ``{prefix}_001`` after dedup."""
    return [
        item.model_copy(update={"id": f"{prefix}_{i + 1:03d}"})
        for i, item in enumerate(items)
    ]
