"""Pydantic v2 models for API I/O and internal pipeline items.

Coordinate convention: all polygons/bboxes are in **original image pixels**,
never tile-local. A polygon is a quad (4 points) ``[[x,y],...]``; an axis-aligned
bbox ``[x1,y1,x2,y2]`` is also carried for convenience (NMS, annotation).
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

ItemType = Literal["text", "art_text", "qr", "barcode"]
ItemSource = Literal["paddleocr", "vlm_fallback", "pyzbar"]


class Item(BaseModel):
    """One detected element (text line, art text, or code)."""

    id: str
    type: ItemType
    # text is used for text/art_text; content for decoded qr/barcode payload
    text: Optional[str] = None
    content: Optional[str] = None
    # quad polygon in original-image coordinates
    polygon: list[list[float]]
    # axis-aligned bbox [x1, y1, x2, y2] (derived from polygon, for NMS/annot)
    bbox: list[float]
    confidence: float = Field(ge=0.0, le=1.0)
    source: ItemSource
    # which tile produced it (None for whole-image); useful for debugging
    tile_index: Optional[int] = None


class ImageMeta(BaseModel):
    width: int
    height: int
    tile_count: int = 1


class AnalyzeResponse(BaseModel):
    image_meta: ImageMeta
    items: list[Item]
    annotated_image_b64: Optional[str] = None
    # populated only for async responses
    task_id: Optional[str] = None


class TaskStatus(BaseModel):
    task_id: str
    status: Literal["pending", "running", "done", "error"]
    result: Optional[AnalyzeResponse] = None
    error: Optional[str] = None
