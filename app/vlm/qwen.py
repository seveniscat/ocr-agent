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
        self._enable_thinking = getattr(settings, "vlm_enable_thinking", False)

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

        text, _conf = self.ask_image(
            b64, _PROMPT, max_tokens=128, json_mode=False
        )
        text = text.strip()
        if not text or text.upper() == "EMPTY":
            return "", 0.0
        # No native score; we model confidence as 1 - (len/pixels heuristic) is
        # unreliable, so we return a flat-but-high value (the VLM only fires when
        # PaddleOCR was already unsure — winning by having *any* read beats none).
        text = re.sub(r"^['\"]|['\"]$", "", text)
        return text, 0.8

    # NOTE: recognize_crops_batch is inherited from VLMProvider, which dispatches
    # the per-crop calls across a thread pool (concurrency) rather than packing
    # them into one multi-image request. Empirically, for art-text fallback on
    # Qwen-VL, parallel single-image calls beat a multi-image call: the latter
    # forces serial cross-image attention server-side and is slower overall.

    def ask_image(
        self,
        image_b64_data_url: str,
        prompt: str,
        *,
        max_tokens: int = 1024,
        json_mode: bool = False,
        enable_thinking: bool | None = None,
        model_override: str | None = None,
    ) -> tuple[str, float]:
        """Send one image + one prompt to Qwen-VL. Returns ``(raw_text, conf)``.

        ``image_b64_data_url`` must be a full ``data:image/...;base64,...`` URL.
        ``json_mode`` sets ``response_format={"type":"json_object"}``.

        ``model_override`` pins a specific model for this call (e.g. the agent
        uses a separate vision model from the art-text fallback). Falls back to
        the provider's configured model when None.

        Thinking mode (Qwen3.x): when enabled, the model reasons before
        answering (``extra_body={"enable_thinking": True}``). Note thinking mode
        is incompatible with ``response_format=json_object`` — when both are
        requested we honor thinking and drop json_mode, relying on the caller's
        tolerant JSON parser instead. Defaults to the provider's setting
        (``self._enable_thinking``) when ``None``.

        Confidence is a flat 0.8 heuristic: Qwen-VL gives no native score, so
        callers should rely on downstream validation (did the JSON parse?) as
        the real quality signal.
        """
        want_thinking = (
            self._enable_thinking if enable_thinking is None else enable_thinking
        )
        # thinking mode + json_object are mutually exclusive on DashScope;
        # when thinking is on, drop json_mode and let the tolerant parser cope.
        use_json = json_mode and not want_thinking

        kwargs: dict = {
            "model": model_override or self._model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_b64_data_url}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "temperature": 0.0,
            "max_tokens": max_tokens,
        }
        if use_json:
            kwargs["response_format"] = {"type": "json_object"}
        if want_thinking:
            # OpenAI-compatible passthrough for the DashScope-specific flag.
            kwargs["extra_body"] = {"enable_thinking": True}
        resp = self._client.chat.completions.create(**kwargs)
        msg = resp.choices[0].message
        text = (getattr(msg, "content", None) or "").strip()
        return text, 0.8


def _to_b64_jpeg(arr) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=92)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"
