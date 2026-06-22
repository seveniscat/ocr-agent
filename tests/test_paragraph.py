"""Tests for paragraph-level merging (line → block) and granularity config.

Pure-geometry tests; no paddle / pyzbar needed.
"""
from __future__ import annotations

import pytest

from app.config import Settings
from app.ocr.detector import (
    TextDetection,
    _quad_bbox,
    _x_overlap_ratio,
    merge_lines_to_paragraphs,
)


def _line(x1, y1, x2, y2, text="x", conf=0.9):
    """Build a line-level TextDetection as an axis-aligned quad."""
    return TextDetection(
        polygon=[[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
        text=text,
        confidence=conf,
        granularity="line",
    )


# ---------------------------------------------------------------------------
# geometry helpers
# ---------------------------------------------------------------------------


def test_quad_bbox():
    poly = [[10, 20], [30, 20], [30, 40], [10, 40]]
    assert _quad_bbox(poly) == (10, 20, 30, 40)


def test_x_overlap_full():
    assert _x_overlap_ratio((0, 100), (20, 80)) == pytest.approx(1.0)


def test_x_overlap_partial():
    # ranges [0,100] and [50,150]; narrower is 100; inter=50 → 0.5
    assert _x_overlap_ratio((0, 100), (50, 150)) == pytest.approx(0.5)


def test_x_overlap_none():
    assert _x_overlap_ratio((0, 100), (200, 300)) == 0.0


# ---------------------------------------------------------------------------
# merge_lines_to_paragraphs
# ---------------------------------------------------------------------------


def test_merge_three_close_lines_into_one_block():
    # Three tightly-stacked lines (gap ~3px, height ~30px) → one paragraph.
    lines = [
        _line(0, 91, 371, 118, "line one"),
        _line(0, 124, 265, 153, "line two"),
        _line(0, 156, 244, 186, "line three"),
    ]
    out = merge_lines_to_paragraphs(lines, gap_ratio=0.6, x_overlap=0.3)
    assert len(out) == 1
    blk = out[0]
    assert blk.granularity == "paragraph"
    # merged quad spans all lines
    assert _quad_bbox(blk.polygon) == (0, 91, 371, 186)
    assert blk.text == "line one\nline two\nline three"
    assert len(blk.lines) == 3


def test_merge_keeps_separate_blocks_on_large_vertical_gap():
    # Two lines far apart vertically (gap >> height) → stay separate.
    lines = [
        _line(0, 0, 100, 30, "title"),
        _line(0, 300, 100, 330, "far below"),
    ]
    out = merge_lines_to_paragraphs(lines, gap_ratio=0.6, x_overlap=0.3)
    assert len(out) == 2
    assert all(b.granularity == "paragraph" for b in out)


def test_merge_requires_x_overlap():
    # Two vertically-close lines but horizontally disjoint (columns) → separate.
    lines = [
        _line(0, 0, 100, 30, "left"),
        _line(500, 5, 600, 30, "right"),  # tiny gap, no x-overlap
    ]
    out = merge_lines_to_paragraphs(lines, gap_ratio=0.6, x_overlap=0.3)
    assert len(out) == 2


def test_merge_mixed_scenario():
    # Realistic packaging layout with generous whitespace between sections.
    # Pure-geometry merging cannot tell "title vs body" apart — it relies on
    # whitespace gaps. So sections here are separated by gaps > gap_ratio*h.
    lines = [
        _line(50, 40, 300, 75, "TITLE"),       # h=35
        _line(50, 120, 371, 150, "desc1"),     # gap 45 (TITLE bottom 75) → split
        _line(50, 152, 265, 180, "desc2"),     # gap 2  → merge
        _line(50, 184, 244, 214, "desc3"),     # gap 4  → merge
        _line(50, 350, 300, 382, "SUBTITLE"),  # gap 136 → split
        _line(50, 460, 262, 490, "body1"),     # gap 78 (>0.6*32) → split
        _line(50, 493, 267, 522, "body2"),     # gap 3  → merge
    ]
    out = merge_lines_to_paragraphs(lines, gap_ratio=0.6, x_overlap=0.3)
    # Expect 4 blocks: TITLE / desc(3) / SUBTITLE / body(2)
    assert len(out) == 4
    counts = [len(b.lines) for b in out]
    assert counts == [1, 3, 1, 2]
    # block texts preserve order
    assert out[1].text == "desc1\ndesc2\ndesc3"


def test_merge_single_line_stands_as_block():
    lines = [_line(0, 0, 100, 30, "solo")]
    out = merge_lines_to_paragraphs(lines, gap_ratio=0.6, x_overlap=0.3)
    assert len(out) == 1
    assert out[0].granularity == "paragraph"
    assert out[0].lines == [lines[0].polygon]


def test_merge_empty():
    assert merge_lines_to_paragraphs([], 0.6, 0.3) == []


def test_confidence_is_mean_of_lines():
    lines = [
        _line(0, 0, 100, 30, conf=0.6),
        _line(0, 5, 100, 30, conf=0.8),
    ]
    out = merge_lines_to_paragraphs(lines, gap_ratio=0.6, x_overlap=0.3)
    assert out[0].confidence == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# config wiring
# ---------------------------------------------------------------------------


def test_config_default_granularity_is_line():
    s = Settings()
    assert s.ocr_granularity == "line"


def test_config_accepts_paragraph_and_word():
    assert Settings(ocr_granularity="paragraph").ocr_granularity == "paragraph"
    assert Settings(ocr_granularity="word").ocr_granularity == "word"


def test_config_paragraph_knobs_in_range():
    s = Settings(ocr_paragraph_gap_ratio=1.0, ocr_paragraph_x_overlap=0.5)
    assert s.ocr_paragraph_gap_ratio == 1.0
    assert s.ocr_paragraph_x_overlap == 0.5


def test_config_detection_tuning_params():
    s = Settings(
        ocr_threshold=0.2,
        ocr_box_thresh=0.5,
        ocr_unclip_ratio=2.5,
        ocr_det_limit_side_len=1280,
    )
    assert s.ocr_threshold == 0.2
    assert s.ocr_box_thresh == 0.5
    assert s.ocr_unclip_ratio == 2.5
    assert s.ocr_det_limit_side_len == 1280
