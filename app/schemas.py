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
Granularity = Literal["word", "line", "paragraph"]


class OCROptions(BaseModel):
    """Per-request overrides for the PaddleOCR engine.

    All fields are optional; ``None`` means "use the server default" (from
    Settings). These are the parameters PaddleOCR's ``predict()`` accepts as
    per-call overrides, so changing them does NOT reload the model — it's
    cheap and instant. Designed for interactive tuning in the Web UI.

    See https://paddlepaddle.github.io/PaddleOCR/ for parameter semantics.
    """

    # ---- DB++ text detector ----
    text_det_thresh: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="像素概率阈值。调低→更多框(召回↑)但噪声↑。艺术字救召回时调到 0.2。",
    )
    text_det_box_thresh: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="框内平均分阈值。调低→保留淡色字/艺术字。",
    )
    text_det_unclip_ratio: Optional[float] = Field(
        None, ge=0.5, le=5.0,
        description="框放大系数。调大→框更松(容纳艺术字溢出笔画)。",
    )
    text_det_limit_side_len: Optional[int] = Field(
        None, ge=320, le=4096,
        description="检测前长边缩放目标。调大→小字更清晰但更耗内存。",
    )
    text_det_limit_type: Optional[Literal["max", "min"]] = Field(
        None, description="max=只缩小保细节; min=只放大提速。",
    )

    # ---- recognizer ----
    text_rec_score_thresh: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="识别置信度门槛。低于此值的框被丢弃。调低→保留更多低质量识别。",
    )
    use_textline_orientation: Optional[bool] = Field(
        None, description="是否做文字行方向分类(旋转/倒置文字)。关闭→更快。",
    )

    # ---- output granularity ----
    granularity: Optional[Granularity] = Field(
        None, description="word=词框 / line=行框(默认) / paragraph=段落块(合并行)。",
    )
    paragraph_gap_ratio: Optional[float] = Field(
        None, ge=0.0, le=3.0,
        description="[paragraph] 行间距 ≤ 此值×行高 才合并。调大→更激进合并。",
    )
    paragraph_x_overlap: Optional[float] = Field(
        None, ge=0.0, le=1.0,
        description="[paragraph] 行水平重叠比例下限。",
    )

    def to_predict_kwargs(self) -> dict:
        """Translate to the kwargs PaddleOCR.predict() accepts (snake_case→its names)."""
        kw: dict = {}
        if self.text_det_thresh is not None:
            kw["text_det_thresh"] = self.text_det_thresh
        if self.text_det_box_thresh is not None:
            kw["text_det_box_thresh"] = self.text_det_box_thresh
        if self.text_det_unclip_ratio is not None:
            kw["text_det_unclip_ratio"] = self.text_det_unclip_ratio
        if self.text_det_limit_side_len is not None:
            kw["text_det_limit_side_len"] = self.text_det_limit_side_len
        if self.text_det_limit_type is not None:
            kw["text_det_limit_type"] = self.text_det_limit_type
        if self.text_rec_score_thresh is not None:
            kw["text_rec_score_thresh"] = self.text_rec_score_thresh
        if self.use_textline_orientation is not None:
            kw["use_textline_orientation"] = self.use_textline_orientation
        return kw


class Item(BaseModel):
    """One detected element (text line/word/paragraph, art text, or code)."""

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
    # output granularity of the text box: word | line | paragraph
    granularity: Optional[Granularity] = None
    # for paragraph granularity: the per-line quads that were merged into this
    # block (kept so the UI / downstream can still draw individual lines)
    lines: Optional[list[list[list[float]]]] = None


class ImageMeta(BaseModel):
    width: int
    height: int
    tile_count: int = 1
    # Original-image coords of the auto-crop box [x0, y0, x1, y1] (exclusive end).
    # None when no crop was applied (feature disabled, or the image was blank).
    # All item polygons/bboxes below are in the *cropped* space, so to map an
    # item back to the original image add crop[0]/crop[1] to its x/y.
    crop: Optional[list[int]] = None


class AnalyzeResponse(BaseModel):
    image_meta: ImageMeta
    items: list[Item]
    # echo back the effective options used (helps the UI know what was applied)
    options_used: Optional[OCROptions] = None
    annotated_image_b64: Optional[str] = None
    # populated only for async responses
    task_id: Optional[str] = None


class TaskStatus(BaseModel):
    task_id: str
    status: Literal["pending", "running", "done", "error"]
    result: Optional[AnalyzeResponse] = None
    error: Optional[str] = None


class WebhookPayload(BaseModel):
    """Outbound webhook body — a tiny status ping sent to a caller-supplied URL.

    Sent from ``/analyze`` when the request carries ``callback_url``. Carries
    only status + task_id (+ the caller's ``biz_id``); the receiver pulls the
    full OCR result via ``GET /tasks/{task_id}``. Constructed in
    ``app/webhook.build_payload``; this model documents the wire contract.
    """

    event: Literal["analyze.completed", "analyze.failed"]
    task_id: str
    status: Literal["done", "error"]
    timestamp: str  # ISO-8601 UTC, e.g. "2026-06-24T12:34:56Z"
    biz_id: Optional[str] = None  # echoed verbatim from the request, when given
    error: Optional[str] = None    # short message; only set on failure


class PanelItem(BaseModel):
    """One main panel cut from a die-line (a box face)."""

    index: int                          # 1-based, sorted top→bottom then left→right
    bbox: list[int]                     # [x0, y0, x1, y1] in original-image px
    width: int
    height: int
    image_b64: Optional[str] = None     # base64 PNG crop (only if preview requested)


class PanelsResponse(BaseModel):
    """Result of ``POST /panels`` — a die-line split into its main faces."""

    width: int   # original image width (px)
    height: int  # original image height (px)
    count: int   # number of panels returned (typically 5 or 6)
    panels: list[PanelItem]
    # The detected main-body bbox [x0,y0,x1,y1] (original px). The equal-split
    # panels subdivide this box. None when no subject was found (blank image).
    subject: Optional[list[int]] = None


# ---------------------------------------------------------------------------
# Interactive panel splitting (auto candidates + user confirmation)
# ---------------------------------------------------------------------------


class CandidateLine(BaseModel):
    """A proposed cut line from ``POST /panels/candidates``.

    ``pos`` is the pixel coordinate along the perpendicular axis (y for
    horizontal lines, x for vertical lines), in original-image px.
    ``confidence`` is one of {0.6 low, 0.8 mid, 1.0 high}. ``selected`` is the
    server's default pre-selection (high-conf only); the user can toggle any
    line in the UI before computing panels.
    """

    pos: int
    orientation: Literal["h", "v"]
    confidence: float
    selected: bool


class CandidatesResponse(BaseModel):
    """Proposed cut lines for the user to confirm in the interactive splitter."""

    width: int
    height: int
    h_lines: list[CandidateLine]   # horizontal lines (each has a y coordinate)
    v_lines: list[CandidateLine]   # vertical lines (each has an x coordinate)


class ComputePanelsRequest(BaseModel):
    """User-confirmed cut lines → ``POST /panels/compute``."""

    h_lines: list[int] = Field(
        default_factory=list,
        description="Confirmed y-coordinates of horizontal cut lines (orig px).",
    )
    v_lines: list[int] = Field(
        default_factory=list,
        description="Confirmed x-coordinates of vertical cut lines (orig px).",
    )
    width: int
    height: int
    image_b64: Optional[str] = Field(
        None,
        description="Base64 image (data URI or raw). When present, each returned "
                    "panel includes a base64 PNG crop. Omit for bbox-only output.",
    )


class ComputePanelsResponse(BaseModel):
    """Panels computed from user-confirmed cut lines."""

    width: int
    height: int
    count: int
    panels: list[PanelItem]


class VlmPanelItem(BaseModel):
    """One panel detected by the VLM splitter."""

    index: int
    bbox: list[int]                     # [x0,y0,x1,y1] original-image px
    width: int
    height: int
    label: str = "unknown"              # front/back/left/right/top/bottom/unknown
    image_b64: Optional[str] = None


class VlmPanelsResponse(BaseModel):
    """Result of ``POST /panels/vlm`` — VLM-identified face boxes."""

    width: int
    height: int
    count: int
    panels: list[VlmPanelItem]
    model: str = ""
    # Populated when VLM/parse failed — UI shows this instead of an empty list.
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# AI Native understanding layer (``POST /understand``)
# ---------------------------------------------------------------------------

# Keep this list in sync with the prompt in app/understanding.py.
ElementKind = Literal[
    "logo",
    "product_image",
    "text_block",
    "barcode",
    "qr",
    "nutrition_table",
    "color_block",
    "other",
]


class KeyElement(BaseModel):
    """One salient visual element the VLM noticed while understanding the image.

    ``location`` is a coarse normalized box ``[x, y, w, h]`` in 0–1 fractions of
    the image. VLMs are unreliable at precise localization, so treat it as a
    "roughly here" hint — never as a pixel-accurate bbox. None when the VLM
    didn't offer a location.
    """

    kind: ElementKind
    description: str
    location: Optional[list[float]] = None


class UnderstandingResult(BaseModel):
    """Structured 'what is this image' answer from the VLM understanding layer.

    This is the first AI-Native output: the VLM has graduated from "art-text
    recognition fallback" to "image understander". Level-1 understanding is a
    single whole-image pass (no per-panel calls, no OCR) — enough to answer
    "what kind of packaging is this" and surface the salient elements.
    """

    category: str                      # e.g. "食品-饮料" / "日化" / "药品" / "其他"
    category_confidence: float = Field(ge=0.0, le=1.0)
    panel_count_estimate: int = Field(
        ge=1, le=20,
        description="VLM's guess at the panel/face count "
                    "(1=单面标签, 6=纸盒展开图).",
    )
    style_keywords: list[str] = Field(
        default_factory=list,
        description='Style descriptors, e.g. ["极简","手绘","高饱和"].',
    )
    dominant_colors: list[str] = Field(
        default_factory=list,
        description='Hex colors, e.g. ["#E63946","#F1FAEE"].',
    )
    key_elements: list[KeyElement] = Field(default_factory=list)
    summary: str = Field("", description="一句话概述。")
    # Populated when JSON parsing/validation failed — the raw VLM output is kept
    # so the UI can show *something* instead of an opaque error. In that case
    # category_confidence is 0.0 and the typed fields are best-effort defaults.
    raw_note: Optional[str] = None
    model: str = ""                    # which VLM produced this
