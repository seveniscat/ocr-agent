"""Tests for the tiling engine — the highest-bug-density module.

These run without paddle / pyzbar installed (pure geometry + numpy + PIL).
"""
from __future__ import annotations

import numpy as np
import pytest

from app.schemas import Item
from app.tiling import (
    bbox_iou,
    crop_tile,
    dedupe_items,
    offset_polygon,
    plan_grid,
    polygon_to_bbox,
    tile_specs,
)


# ---------------------------------------------------------------------------
# Grid planning
# ---------------------------------------------------------------------------


def test_sub_4000_long_edge_single_tile():
    """v1 scope: long edge ≤ 4000 → one tile (direct predict, no seam dedupe)."""
    grid = plan_grid(3500, 2800, target_size=3500, overlap=0.15)
    assert grid.count == 1
    assert not grid.needs_tiling


def test_small_image_single_tile():
    grid = plan_grid(1200, 800, target_size=4000, overlap=0.15)
    assert grid.count == 1
    assert grid.needs_tiling is False
    specs = tile_specs(grid)
    assert len(specs) == 1
    s = specs[0]
    assert (s.x0, s.y0, s.x1, s.y1) == (0, 0, 1200, 800)


def test_tiles_cover_whole_image_no_gaps():
    """Critical invariant: every pixel must be covered by at least one tile."""
    w, h = 8000, 6000
    grid = plan_grid(w, h, target_size=1600, overlap=0.15)
    specs = tile_specs(grid)
    assert grid.needs_tiling

    covered = np.zeros((h, w), dtype=bool)
    for s in specs:
        assert s.x1 <= w and s.y1 <= h
        assert s.x0 < s.x1 and s.y0 < s.y1
        covered[s.y0:s.y1, s.x0:s.x1] = True
    assert covered.all(), "Some pixels are not covered by any tile"


def test_tiles_flush_with_edges():
    """Last column/row of tiles must touch the right/bottom edge."""
    w, h = 8000, 6000
    grid = plan_grid(w, h, target_size=1600, overlap=0.15)
    specs = tile_specs(grid)
    assert max(s.x1 for s in specs) == w
    assert max(s.y1 for s in specs) == h
    # And at least one tile starts at 0,0.
    assert any(s.x0 == 0 and s.y0 == 0 for s in specs)


def test_overlap_between_neighbours():
    """Adjacent tiles must overlap (otherwise glyphs on the seam get split)."""
    grid = plan_grid(5000, 4000, target_size=1600, overlap=0.2)
    specs = tile_specs(grid)
    xs = sorted({s.x0 for s in specs})
    # For any two consecutive origins, the previous tile extends past the next origin.
    for a, b in zip(xs, xs[1:]):
        assert a + grid.tile_w > b, f"No horizontal overlap between x={a} and x={b}"


def test_origin_count_for_known_size():
    """Sanity check on number of tiles."""
    grid = plan_grid(3200, 1600, target_size=1600, overlap=0.15)
    # width 3200 needs >1 column; height 1600 fits in one row.
    assert grid.cols >= 2
    assert grid.rows >= 1


# ---------------------------------------------------------------------------
# Coordinate remap
# ---------------------------------------------------------------------------


def test_offset_polygon_maps_local_to_global():
    poly = [[10, 20], [30, 20], [30, 40], [10, 40]]
    g = offset_polygon(poly, dx=1000, dy=2000)
    assert g == [[1010, 2020], [1030, 2020], [1030, 2040], [1010, 2040]]


def test_round_trip_via_crop_and_offset():
    """End-to-end sanity: put a marker in the image, crop a tile, read it back."""
    img = np.zeros((3000, 4000, 3), dtype="uint8")
    # Draw a 10x10 white square at global (1500, 1200).
    img[1200:1210, 1500:1510] = 255

    grid = plan_grid(4000, 3000, target_size=1600, overlap=0.15)
    specs = tile_specs(grid)
    # Find the tile that contains the marker.
    hit = [s for s in specs if s.x0 <= 1500 < s.x1 and s.y0 <= 1200 < s.y1]
    assert hit, "Marker not covered by any tile"
    s = hit[0]
    tile = crop_tile(img, s)
    # Local coords = global - offset.
    lx, ly = 1500 - s.x0, 1200 - s.y0
    assert (tile[ly:ly + 10, lx:lx + 10] == 255).all()


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def test_polygon_to_bbox():
    poly = [[10, 20], [30, 20], [30, 40], [10, 40]]
    assert polygon_to_bbox(poly) == [10, 20, 30, 40]


def test_bbox_iou_basic():
    assert bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]) == pytest.approx(1.0)
    assert bbox_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    # Half overlap: 5x5 inter over (100 + 100 - 25) = 25/175
    iou = bbox_iou([0, 0, 10, 10], [5, 5, 15, 15])
    assert iou == pytest.approx(25 / 175, rel=1e-6)


# ---------------------------------------------------------------------------
# Dedupe / NMS
# ---------------------------------------------------------------------------


def _item(item_id: str, text: str, bbox, conf=0.9, source="paddleocr", type_="text"):
    return Item(
        id=item_id,
        type=type_,
        text=text,
        polygon=[[bbox[0], bbox[1]], [bbox[2], bbox[1]],
                 [bbox[2], bbox[3]], [bbox[0], bbox[3]]],
        bbox=bbox,
        confidence=conf,
        source=source,
    )


def test_dedupe_merges_high_iou_same_text():
    items = [
        _item("a", "净含量：250ml", [100, 100, 300, 140], conf=0.95),
        _item("b", "净含量：250ml", [102, 101, 302, 141], conf=0.80),
    ]
    out = dedupe_items(items)
    assert len(out) == 1
    # Higher-confidence item survives.
    assert out[0].confidence == 0.95


def test_dedupe_keeps_high_iou_different_text():
    # Two overlapping boxes but clearly different text → keep both.
    items = [
        _item("a", "净含量：250ml", [100, 100, 300, 140], conf=0.95),
        _item("b", "生产日期：2025", [102, 101, 302, 141], conf=0.93),
    ]
    out = dedupe_items(items)
    assert len(out) == 2


def test_deduke_no_text_items_merge_by_geometry():
    items = [
        _item("a", None, [100, 100, 200, 200], conf=1.0, type_="qr"),
        _item("b", None, [101, 100, 201, 200], conf=1.0, type_="qr"),
    ]
    out = dedupe_items(items)
    assert len(out) == 1


def test_dedupe_disjoint_kept():
    items = [
        _item("a", "hello", [0, 0, 100, 40]),
        _item("b", "world", [500, 500, 600, 540]),
    ]
    assert len(dedupe_items(items)) == 2


def test_dedupe_drops_tile_seam_text_fragment():
    """Full line + trailing fragment at tile seam → keep the longer text."""
    full = (
        "Nessun personaggio duplicato all'interno della scatola completa."
    )
    items = [
        _item("full", full, [547, 830, 1107, 852], conf=0.998),
        _item("frag", "completa.", [1017, 827, 1110, 855], conf=0.999),
    ]
    out = dedupe_items(items)
    assert len(out) == 1
    assert out[0].text == full


def test_dedupe_official_seam_merge_same_text():
    """Adjacent seam boxes within merge_x/y thres with same text → one item."""
    items = [
        _item("a", "hello world", [100, 100, 200, 130], conf=0.9),
        _item("b", "hello world", [198, 101, 280, 131], conf=0.85),
    ]
    out = dedupe_items(items, merge_x_thres=50, merge_y_thres=35)
    assert len(out) == 1
