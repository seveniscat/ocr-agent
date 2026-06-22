"""Qwen-VL (OpenAI-compatible) provider for art-text fallback.

DashScope exposes an OpenAI-compatible endpoint; we send the cropped region as
a base64 image with a short, strict prompt asking for the literal text only.
This avoids the "VLM emits coordinates" problem entirely — we just want chars.
"""
from __future__ import annotations

import base64
import io
import logging
import re

from ..config import Settings
from .base import VLMProvider

logger = logging.getLogger(__name__)

_PROMPT = (
    "Read out the text visible in this image, exactly as written. "
    "Output ONLY the recognized text, no commentary, no quotes. "
    "If the image contains no readable text, output the single word: EMPTY"
)


class QwenVLM(VLMProvider):
    name = "qwen"

    def __init__(self, settings: Settings) -> None:
        from openai import OpenAI  # lazy import

        if not settings.vlm_api_key:
            raise RuntimeError(
                "OCR_VLM_API_KEY is not set; VLM fallback disabled. "
                "Either set the key in .env or set OCR_VLM_ENABLED=false."
            )
        self._client = OpenAI(
            api_key=settings.vlm_api_key,
            base_url=settings.vlm_base_url,
        )
        self._model = settings.vlm_model

    def recognize_crop(self, image, polygon) -> tuple[str, float]:
        # Tight bbox around the quad (art text on dielines is mostly axis-aligned).
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
        h, w = image.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return "", 0.0

        crop = image[y1:y2, x1:x2]
        b64 = _to_b64_jpeg(crop)

        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": b64}},
                        {"type": "text", "text": _PROMPT},
                    ],
                }
            ],
            temperature=0.0,
            max_tokens=128,
        )
        text = (resp.choices[0].message.content or "").strip()
        if not text or text.upper() == "EMPTY":
            return "", 0.0
        # No native score; we model confidence as 1 - (len/pixels heuristic) is
        # unreliable, so we return a flat-but-high value (the VLM only fires when
        # PaddleOCR was already unsure — winning by having *any* read beats none).
        text = re.sub(r"^['\"]|['\"]$", "", text)
        return text, 0.8


def _to_b64_jpeg(arr) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=92)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"
