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
from ..tiling import dedupe_items, renumber


def aggregate(items: list[Item], prefix: str = "t") -> list[Item]:
    """Deduplicate and renumber a flat list of globally-coordinated items."""
    return renumber(dedupe_items(items), prefix=prefix)
