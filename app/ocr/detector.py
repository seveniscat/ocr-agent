"""PaddleOCR wrapper: polygon detection + recognition per tile.

Targets **PaddleOCR 3.x** (PP-OCRv6 pipeline). API notes:
- Constructor uses the 3.x param names: ``use_textline_orientation``,
  ``text_det_thresh``, ``text_det_unclip_ratio``.
- ``predict(img)`` returns a list whose first element is an ``OCRResult`` (a
  dict subclass). We read three parallel lists per detection:
  ``rec_polys`` (quad), ``rec_texts`` (str), ``rec_scores`` (float).
- Per-call override: ``predict`` accepts the same det/rec params, so we can
  lower ``text_det_thresh`` / raise ``text_det_unclip_ratio`` on suspect tiles
  to fish back art-text recall (the L1 knob — see project plan).

PaddleOCR's result polygons are in the input tile's coordinate space; the
pipeline offsets them to global coords. The model loads lazily on first call
so ``import app.main`` stays fast and tests can mock the engine.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..config import Settings

logger = logging.getLogger(__name__)


@dataclass
class TextDetection:
    """One recognized text line within a tile (tile-local coords)."""

    polygon: list[list[float]]  # quad [[x,y],...]
    text: str
    confidence: float


def _to_numpy(tile) -> np.ndarray:
    """Accept numpy array or PIL image; return HxWx3 uint8 RGB."""
    if isinstance(tile, np.ndarray):
        return tile
    # PIL image
    return np.array(tile.convert("RGB"))


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

        logger.info("Loading PaddleOCR 3.x (PP-OCRv6)…")
        # 3.x API. lang 'ch' covers Chinese + English (packaging multilingual).
        self._ocr = PaddleOCR(
            lang="ch",
            use_doc_orientation_classify=False,  # dielines are upright; skip
            use_doc_unwarping=False,             # we tile ourselves; skip
            use_textline_orientation=True,       # rotated text on packaging
            text_det_thresh=self.settings.ocr_threshold,
            # Slightly inflate unclip to widen boxes — helps catch art text
            # whose glyph bounds are larger than the shrink-map suggests.
            text_det_unclip_ratio=1.8,
        )
        logger.info("PaddleOCR ready.")

    def detect_and_recognize(
        self, tile, *, det_thresh: float | None = None
    ) -> list[TextDetection]:
        """Run full OCR on a tile (HxWx3 numpy uint8 or PIL). Tile-local dets.

        ``det_thresh`` optionally overrides the configured detection threshold
        for this single call — used to fish back art-text recall on suspect
        tiles without reconfiguring the engine.
        """
        self._ensure_loaded()
        arr = _to_numpy(tile)

        kwargs: dict[str, Any] = {}
        if det_thresh is not None:
            kwargs["text_det_thresh"] = det_thresh

        results = self._ocr.predict(arr, **kwargs)
        if not results:
            return []

        # predict returns a list (one entry per input page); we feed one page.
        r0 = results[0]
        polys = r0.get("rec_polys") or []
        texts = r0.get("rec_texts") or []
        scores = r0.get("rec_scores") or []
        n = min(len(polys), len(texts), len(scores))
        if n == 0:
            return []

        out: list[TextDetection] = []
        for i in range(n):
            poly = polys[i]
            if poly is None or len(poly) < 4:
                continue
            poly_list = [[float(p[0]), float(p[1])] for p in poly[:4]]
            out.append(
                TextDetection(
                    polygon=poly_list,
                    text=str(texts[i]),
                    confidence=float(scores[i]),
                )
            )
        return out
