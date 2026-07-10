"""QR / barcode detection via pyzbar (zbar under the hood).

pyzbar is fast, local, and returns decoded payloads + quad polygons — exactly
what we need. On huge images pyzbar is invoked per-tile by the pipeline (see
:mod:`app.pipeline`); on small images the pipeline calls it on the whole image.

For curved / artistic QR codes that pyzbar misses, a WeChat QRCode fallback is
the conventional next step (v2 — interface preserved here).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

logger = logging.getLogger(__name__)

CodeType = Literal["qr", "barcode"]


@dataclass
class CodeDetection:
    """A decoded QR/barcode within a tile (tile-local coords)."""

    type: CodeType
    content: str
    polygon: list[list[float]]
    confidence: float = 1.0  # pyzbar gives no per-code score; decoded==confident


# pyzbar type string -> our CodeType
_TYPE_MAP = {
    "QRCODE": "qr",
    "BARCODE": "barcode",
}


def _poly_bbox(poly: list[list[float]]) -> tuple[float, float, float, float]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def _is_duplicate_code(
    cand: CodeDetection, existing: list[CodeDetection]
) -> bool:
    """True when ``cand`` is the same code as one already in ``existing``.

    Same content AND overlapping polygon (the upscaled re-detection of a code
    that also decoded at native scale). Different content at the same spot is
    kept (rare, but shouldn't be silently dropped).
    """
    cx1, cy1, cx2, cy2 = _poly_bbox(cand.polygon)
    for e in existing:
        if e.content != cand.content:
            continue
        ex1, ey1, ex2, ey2 = _poly_bbox(e.polygon)
        ix1, iy1 = max(cx1, ex1), max(cy1, ey1)
        ix2, iy2 = min(cx2, ex2), min(cy2, ey2)
        iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
        if iw > 0 and ih > 0:
            return True
    return False


class CodeEngine:
    """Wraps pyzbar. Lazy-loaded so import is cheap."""

    def __init__(self) -> None:
        self._ready = False

    def _ensure_ready(self) -> None:
        if self._ready:
            return
        try:
            import pyzbar.pyzbar as _  # noqa: F401 — probe import
        except ImportError as exc:  # pragma: no cover — env-dependent
            raise RuntimeError(
                "pyzbar is not installed, or the zbar system library is missing. "
                "Run: pip install pyzbar  AND  brew install zbar (macOS) / "
                "apt-get install libzbar0 (Debian)."
            ) from exc
        self._ready = True

    def available(self) -> bool:
        """Probe the zbar lib once. Returns False if unavailable.

        Lets the pipeline decide up front whether to even try the code channel,
        rather than failing mid-request on the first tile.
        """
        try:
            self._ensure_ready()
            return True
        except Exception:  # noqa: BLE001
            return False

    def detect(self, tile) -> list[CodeDetection]:
        """Detect + decode all codes in a tile (HxWx3 numpy uint8).

        Tries the image at native scale first (fast), then re-tries at 2× and
        3× upscale. Small QR codes — common on packaging artwork downscaled to
        a screenshot — fall below zbar's decodable module size at native scale
        but decode cleanly once upscaled. Upscaled hits are de-duplicated against
        native ones by content + polygon overlap, and their coordinates are
        scaled back to the tile's native space.
        """
        self._ensure_ready()
        from PIL import Image

        import numpy as _np

        # native scale
        out = self._decode_at(Image.fromarray(tile), scale=1.0)
        if not out:
            return []

        # If the tile is small, small QRs likely failed at native scale — retry
        # upscaled and merge any NEW codes (by content + overlap) back in.
        h, w = tile.shape[:2]
        if min(h, w) < 1500:
            for factor in (2, 3):
                big = Image.fromarray(tile).resize(
                    (w * factor, h * factor), Image.LANCZOS
                )
                upscaled = self._decode_at(big, scale=1.0 / factor)
                for d in upscaled:
                    if not _is_duplicate_code(d, out):
                        out.append(d)
        return out

    def _decode_at(self, pil, *, scale: float) -> list[CodeDetection]:
        """Decode codes from a PIL image, scaling polygon coords by ``scale``."""
        from pyzbar.pyzbar import decode

        results = decode(pil)
        out: list[CodeDetection] = []
        for r in results:
            raw_type = (r.type or "").upper()
            code_type = _TYPE_MAP.get(raw_type, "barcode")
            # pyzbar's `rect` is axis-aligned; `polygon` is the quad (preferred).
            if r.polygon:
                poly = [[float(p[0]) * scale, float(p[1]) * scale]
                        for p in r.polygon]
            else:
                left = r.rect.left * scale
                top = r.rect.top * scale
                poly = [
                    [left, top],
                    [r.rect.width * scale + left, top],
                    [r.rect.width * scale + left, r.rect.height * scale + top],
                    [left, r.rect.height * scale + top],
                ]
            content = r.data.decode("utf-8", errors="replace")
            out.append(
                CodeDetection(type=code_type, content=content, polygon=poly)
            )
        return out
