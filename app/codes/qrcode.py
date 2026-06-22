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
        """Detect + decode all codes in a tile (HxWx3 numpy uint8)."""
        self._ensure_ready()
        from pyzbar.pyzbar import decode
        from PIL import Image

        # pyzbar accepts PIL or raw; PIL path is the most robust across builds.
        pil = Image.fromarray(tile)
        results = decode(pil)
        out: list[CodeDetection] = []
        for r in results:
            raw_type = (r.type or "").upper()
            code_type = _TYPE_MAP.get(raw_type, "barcode")
            # pyzbar's `rect` is axis-aligned; `polygon` is the quad (preferred).
            poly = (
                [[float(p[0]), float(p[1])] for p in r.polygon]
                if r.polygon
                else [
                    [float(r.rect.left), float(r.rect.top)],
                    [float(r.rect.left + r.rect.width), float(r.rect.top)],
                    [float(r.rect.left + r.rect.width),
                     float(r.rect.top + r.rect.height)],
                    [float(r.rect.left), float(r.rect.top + r.rect.height)],
                ]
            )
            content = r.data.decode("utf-8", errors="replace")
            out.append(
                CodeDetection(
                    type=code_type,
                    content=content,
                    polygon=poly,
                )
            )
        return out
