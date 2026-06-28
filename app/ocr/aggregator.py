"""Cross-tile aggregation.

The heavy lifting (polygon offset + NMS + text-similarity merge) lives in
:mod:`app.tiling`, because it operates on generic :class:`Item` objects and is
reusable. This module is the thin adapter that turns raw per-tile detections
into globally-coordinated :class:`Item` objects and calls into tiling.

Kept separate so future aggregation strategies (e.g. panel-aware merging on
packaging dielines) have a home.
"""
from __future__ import annotations

from ..schemas import Item
from ..tiling import dedupe_items, polygon_to_bbox, renumber
from .detector import TextDetection, merge_lines_to_paragraphs


def apply_paragraph_granularity(
    items: list[Item],
    gap_ratio: float,
    x_overlap: float,
) -> list[Item]:
    """Merge line-level PaddleOCR text items into paragraph blocks (global coords).

    Paragraph merging used to run per tile only, so adjacent lines split across
    tile seams never combined and the UI looked identical to ``line`` mode.
    We always collect line boxes first, then merge once in image space.
    """
    texts = [
        it for it in items if it.type == "text" and it.source == "paddleocr"
    ]
    others = [
        it for it in items if not (it.type == "text" and it.source == "paddleocr")
    ]
    if not texts:
        return items

    line_dets = [
        TextDetection(
            polygon=it.polygon,
            text=it.text or "",
            confidence=it.confidence,
            granularity="line",
        )
        for it in texts
    ]
    merged = merge_lines_to_paragraphs(line_dets, gap_ratio, x_overlap)
    new_texts = [
        Item(
            id="tmp",
            type="text",
            text=m.text,
            polygon=m.polygon,
            bbox=polygon_to_bbox(m.polygon),
            confidence=m.confidence,
            source="paddleocr",
            tile_index=None,
            granularity="paragraph",
            lines=m.lines,
        )
        for m in merged
    ]
    return new_texts + others


def aggregate(items: list[Item], prefix: str = "t") -> list[Item]:
    """Deduplicate and renumber a flat list of globally-coordinated items."""
    return renumber(dedupe_items(items), prefix=prefix)
