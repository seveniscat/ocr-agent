"""PaddleOCR wrapper: polygon detection + recognition per tile.

Targets **PaddleOCR 3.x** with **PP-OCRv6** by default
(``PP-OCRv6_medium_det`` + ``PP-OCRv6_medium_rec``, single-model ≈50-language rec).
Low-confidence crops are re-read by the VLM only when
``vlm_ocr_fallback_enabled`` is set (opt-in; default off).

Output granularity (controlled by ``Settings.ocr_granularity``):
- ``word``      — per-token boxes. Uses PaddleOCR's native ``return_word_box``.
- ``line``      — default text-line boxes (most common; PaddleOCR default).
- ``paragraph`` — line boxes grouped into paragraph blocks by geometric
                  proximity. Implemented as a post-processing step here (no
                  extra model needed): two lines merge when they are vertically
                  close (gap <= k·line_height) and horizontally overlapping.

Detection tuning (DB++): ``text_det_thresh`` / ``text_det_box_thresh`` /
``text_det_unclip_ratio`` / ``text_det_limit_side_len`` are forwarded to
PaddleOCR and can also be overridden per-call (used to fish back art-text
recall on suspect tiles).

PaddleOCR's result polygons are in the input tile's coordinate space; the
pipeline offsets them to global coords. The model loads lazily on first call
so ``import app.main`` stays fast and tests can mock the engine.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..config import Granularity, Settings

logger = logging.getLogger(__name__)


@dataclass
class TextDetection:
    """One recognized text unit within a tile (tile-local coords).

    ``polygon`` is the *quad* actually drawn; when granularity is ``paragraph``
    this is the merged block quad, and ``lines`` carries the per-line quads
    that were merged into it.

    ``recognized`` is False for boxes the detector found but the recognizer
    dropped (rec score below threshold — almost always a script the loaded rec
    model can't read, e.g. Korean under PP-OCRv6). For these ``text`` is empty
    and ``confidence`` is 0; ``crop_b64`` carries a base64 PNG of the region so
    a downstream script-aware model can re-read it.
    """

    polygon: list[list[float]]  # quad [[x,y],...]
    text: str
    confidence: float
    granularity: Granularity = "line"
    # populated only in paragraph mode: the per-line quads merged here
    lines: list[list[list[float]]] | None = None
    # True = recognition succeeded; False = detector-only (awaiting external rec)
    recognized: bool = True
    # base64 PNG crop, populated for unrecognized boxes when emit_crops is on
    crop_b64: str | None = None


def _to_numpy(tile) -> np.ndarray:
    """Accept numpy array or PIL image; return HxWx3 uint8 RGB."""
    if isinstance(tile, np.ndarray):
        return tile
    return np.array(tile.convert("RGB"))


def _encode_crop(tile: np.ndarray, polygon: list[list[float]]) -> str | None:
    """Crop the bbox around ``polygon`` from ``tile`` and return a base64 PNG.

    Returns None for empty/degenerate crops (never raises). Reuses the
    axis-aligned crop from :func:`recognizer.crop_polygon`.
    """
    from .recognizer import crop_polygon

    crop = crop_polygon(tile, polygon)
    if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
        return None
    import base64
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(crop).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# Geometry helpers for paragraph merging (pure functions — unit-testable)
# ---------------------------------------------------------------------------


def _quad_bbox(poly):
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def _x_overlap_ratio(a, b) -> float:
    """Overlap ratio of two x-ranges [ax1,ax2] and [bx1,bx2] over the smaller width."""
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    if inter == 0.0:
        return 0.0
    min_w = min(a[1] - a[0], b[1] - b[0])
    return inter / min_w if min_w > 0 else 0.0


def _same_text_column(a, b, x_overlap: float) -> bool:
    """True when two lines plausibly belong to one copy block (same column).

    Packaging paragraphs are often left-aligned with varying line lengths, so
    strict x-range overlap rejects valid merges. Fall back to left-edge or
    center alignment when vertical gap is already small.
    """
    if _x_overlap_ratio(a, b) >= x_overlap:
        return True
    wa, wb = a[1] - a[0], b[1] - b[0]
    avg_w = (wa + wb) / 2 or 1.0
    # Left-aligned block (common on dielines).
    if abs(a[0] - b[0]) <= 0.12 * avg_w:
        return True
    # Center-aligned multi-line titles.
    ca, cb = (a[0] + a[1]) / 2, (b[0] + b[1]) / 2
    if abs(ca - cb) <= 0.15 * avg_w:
        return True
    return False


def merge_lines_to_paragraphs(
    line_dets: list[TextDetection],
    gap_ratio: float,
    x_overlap: float,
) -> list[TextDetection]:
    """Group line-level detections into paragraph blocks by proximity.

    Two adjacent lines merge when:
      - vertical gap between them <= ``gap_ratio * average_line_height``, AND
      - they sit in the same text column (x-overlap OR left/center alignment).
    A line that matches nothing stays its own (single-line) block.

    Lines are first sorted top-to-bottom; we sweep once, so complexity is O(n).
    This is a deliberate, transparent heuristic — packaging copy blocks are
    mostly axis-aligned, so geometric grouping is robust here. For truly
    complex layouts (multi-column), a layout-detection model (PP-StructureV3)
    would be needed; that path is left as a future ``region`` granularity.
    """
    if len(line_dets) <= 1:
        for d in line_dets:
            d.granularity = "paragraph"
            d.lines = [d.polygon]
        return line_dets

    # Sort by top-y (stable for equal tops to preserve reading order).
    decorated = sorted(
        enumerate(line_dets),
        key=lambda t: (_quad_bbox(t[1].polygon)[1], t[0]),
    )

    groups: list[list[TextDetection]] = []
    current: list[TextDetection] = [decorated[0][1]]
    # Track the previous line's geometry (the last line appended to `current`).
    prev = decorated[0][1]
    px1, py1, px2, py2 = _quad_bbox(prev.polygon)
    prev_bottom, prev_height, prev_xrange = py2, py2 - py1, (px1, px2)

    for _, det in decorated[1:]:
        x1, y1, x2, y2 = _quad_bbox(det.polygon)
        gap = y1 - prev_bottom
        this_height = y2 - y1
        avg_h = (prev_height + this_height) / 2 or 1.0
        this_xrange = (x1, x2)
        if gap <= gap_ratio * avg_h and _same_text_column(
            prev_xrange, this_xrange, x_overlap
        ):
            current.append(det)
        else:
            groups.append(current)
            current = [det]
        # update prev* to THIS line regardless (next comparison is vs this one)
        prev_bottom, prev_height, prev_xrange = y2, this_height, this_xrange
    groups.append(current)

    # Build one TextDetection per group: merged quad = bbox of all lines.
    out: list[TextDetection] = []
    for grp in groups:
        xs = [p[0] for d in grp for p in d.polygon]
        ys = [p[1] for d in grp for p in d.polygon]
        x_min, y_min, x_max, y_max = min(xs), min(ys), max(xs), max(ys)
        merged_quad = [
            [x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]
        ]
        text = "\n".join(d.text for d in grp)
        # block confidence = mean of line confidences
        conf = sum(d.confidence for d in grp) / len(grp)
        out.append(
            TextDetection(
                polygon=merged_quad,
                text=text,
                confidence=conf,
                granularity="paragraph",
                lines=[d.polygon for d in grp],
            )
        )
    return out


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class OCREngine:
    """Wraps PaddleOCR 3.x. Lazy-loaded; safe to construct without paddle."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._ocr: Any = None

    def _ensure_loaded(self) -> None:
        if self._ocr is not None:
            return
        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:  # pragma: no cover — env-dependent
            raise RuntimeError(
                "paddleocr is not installed. "
                "Run: pip install paddleocr paddlepaddle"
            ) from exc

        s = self.settings
        logger.info(
            "Loading PaddleOCR 3.x (version=%s, lang=%s)…",
            s.ocr_version, s.ocr_lang,
        )
        self._ocr = PaddleOCR(
            lang=s.ocr_lang,
            ocr_version=s.ocr_version,
            use_doc_orientation_classify=False,  # dielines are upright
            use_doc_unwarping=False,             # we tile ourselves
            use_textline_orientation=True,       # rotated text on packaging
            # DB++ detection tuning
            text_det_thresh=s.ocr_threshold,
            text_det_box_thresh=s.ocr_box_thresh,
            text_det_unclip_ratio=s.ocr_unclip_ratio,
            text_det_limit_side_len=s.ocr_det_limit_side_len,
            text_det_limit_type=s.ocr_det_limit_type,
            # Per-token boxes (used only when granularity == "word")
            return_word_box=(s.ocr_granularity == "word"),
        )
        logger.info(
            "PaddleOCR ready: version=%s lang=%s granularity=%s "
            "det_thresh=%.2f unclip=%.1f",
            s.ocr_version, s.ocr_lang, s.ocr_granularity,
            s.ocr_threshold, s.ocr_unclip_ratio,
        )

    def detect_and_recognize(
        self,
        tile,
        *,
        det_thresh: float | None = None,
        predict_kwargs: dict | None = None,
        granularity: Granularity | None = None,
        paragraph_gap_ratio: float | None = None,
        paragraph_x_overlap: float | None = None,
        emit_crops: bool | None = None,
    ) -> list[TextDetection]:
        """Run full OCR on a tile (HxWx3 numpy uint8 or PIL). Tile-local dets.

        Tuning overrides (all optional, all cheap — no model reload):
        - ``predict_kwargs``: forwarded verbatim to ``PaddleOCR.predict()``.
          Covers text_det_thresh / text_det_box_thresh / text_det_unclip_ratio /
          text_det_limit_side_len / text_det_limit_type / text_rec_score_thresh /
          use_textline_orientation.
        - ``det_thresh``: shortcut for the most common knob (merged into kwargs).
        - ``granularity`` / paragraph params: control output box level.
        - ``emit_crops``: when True, detector boxes the recognizer dropped are
          kept with ``recognized=False`` and a base64 PNG crop. Defaults to
          ``settings.ocr_emit_crops``. This is the mechanism that stops Korean
          (and other scripts missing from the rec dict) from being silently
          lost: the detector boxes them, the recognizer can't read them, but
          we no longer discard the box.
        """
        self._ensure_loaded()
        arr = _to_numpy(tile)

        kwargs: dict[str, Any] = dict(predict_kwargs or {})
        if det_thresh is not None and "text_det_thresh" not in kwargs:
            kwargs["text_det_thresh"] = det_thresh

        results = self._ocr.predict(arr, **kwargs)
        if not results:
            return []

        r0 = results[0]
        polys = r0.get("rec_polys") or []
        texts = r0.get("rec_texts") or []
        scores = r0.get("rec_scores") or []
        n = min(len(polys), len(texts), len(scores))

        gran = granularity or self.settings.ocr_granularity

        # ---- recognized detections (passed the rec score threshold) ----
        out_label = "word" if gran == "word" else "line"
        line_dets: list[TextDetection] = []
        for i in range(n):
            poly = polys[i]
            if poly is None or len(poly) < 4:
                continue
            poly_list = [[float(p[0]), float(p[1])] for p in poly[:4]]
            line_dets.append(
                TextDetection(
                    polygon=poly_list,
                    text=str(texts[i]),
                    confidence=float(scores[i]),
                    granularity=out_label,
                )
            )

        # ---- detector-only detections: boxes the recognizer dropped ----
        # The recognizer filters out low-score boxes (e.g. Korean text under
        # PP-OCRv6, whose dict has no Hangul). Those boxes are still in the
        # detector's `dt_polys`. Recover them as recognized=False so they're not
        # lost, optionally with a crop for an external model to re-read.
        do_crops = (emit_crops if emit_crops is not None
                    else self.settings.ocr_emit_crops)
        if gran != "paragraph":
            extra = self._recover_unrecognized(r0, arr, line_dets, emit=do_crops)
            line_dets.extend(extra)

        if gran == "paragraph":
            return merge_lines_to_paragraphs(
                line_dets,
                gap_ratio=paragraph_gap_ratio
                if paragraph_gap_ratio is not None
                else self.settings.ocr_paragraph_gap_ratio,
                x_overlap=paragraph_x_overlap
                if paragraph_x_overlap is not None
                else self.settings.ocr_paragraph_x_overlap,
            )
        return line_dets

    def _recover_unrecognized(
        self,
        result: Any,
        tile_arr: np.ndarray,
        recognized: list[TextDetection],
        *,
        emit: bool,
    ) -> list[TextDetection]:
        """Boxes present in ``dt_polys`` but not in ``rec_polys``.

        Matches detector boxes against the already-recognized ones by bbox IoU
        (a dt box that overlaps a rec box ≥ 0.5 IoU was already recognized).
        The unmatched dt boxes are returned as ``recognized=False`` detections,
        each carrying a base64 PNG crop when ``emit`` is True.
        """
        dt_polys = result.get("dt_polys") or []
        if not dt_polys:
            return []

        rec_bboxes = [_quad_bbox(d.polygon) for d in recognized]
        rec_bbox_list = [[b[0], b[1], b[2], b[3]] for b in rec_bboxes]

        from ..tiling import bbox_iou  # local import avoids cycle at module load

        out: list[TextDetection] = []
        for poly in dt_polys:
            if poly is None or len(poly) < 4:
                continue
            poly_list = [[float(p[0]), float(p[1])] for p in poly[:4]]
            x1, y1, x2, y2 = _quad_bbox(poly_list)
            dt_bbox = [x1, y1, x2, y2]
            # Skip dt boxes already covered by a recognized box.
            if any(bbox_iou(dt_bbox, rb) >= 0.5 for rb in rec_bbox_list):
                continue
            crop_b64 = None
            if emit:
                crop_b64 = _encode_crop(tile_arr, poly_list)
            out.append(
                TextDetection(
                    polygon=poly_list,
                    text="",
                    confidence=0.0,
                    granularity="line",
                    recognized=False,
                    crop_b64=crop_b64,
                )
            )
        return out
