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
    "Output the recognized text followed by '||' and your confidence "
    "score (0.0-1.0) that the text is correct. "
    "Format: <text>||<score>  (e.g. 12345||0.95). "
    "No quotes, no commentary. "
    "If the image contains no readable text, output the single word: EMPTY"
)

# Separator between text and the model's self-rated confidence score.
_CONF_SEP = "||"
# Fallback confidence when the model doesn't emit the expected "||score" suffix.
# Kept high enough to survive rec_confidence_drop (0.60) so a format regression
# never silently drops a valid read, but not 1.0 — it's an unknown, not a sure.
_FALLBACK_CONF = 0.8


def _parse_self_rated(text: str) -> tuple[str, float]:
    """Split a ``text||confidence`` VLM response into ``(clean_text, score)``.

    The fallback prompt asks the model to append ``||<0-1>`` to its read. When
    it does, we return the score (clamped to [0, 1]). When it doesn't follow
    the format — or the score is malformed / out of range — we fall back to
    ``_FALLBACK_CONF`` (0.8), matching the pre-change behavior so a format
    regression can't quietly filter out good results.

    ``rpartition`` is used so a text body that itself contains ``||`` keeps its
    earlier occurrences (only the final ``||<score>`` is split off).
    """
    if _CONF_SEP in text:
        body, _, tail = text.rpartition(_CONF_SEP)
        body = body.strip()
        try:
            score = float(tail.strip())
            if 0.0 <= score <= 1.0:
                return body, score
        except ValueError:
            pass
    return text, _FALLBACK_CONF


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
        # Honored by VLMProvider._crop_bbox_or_none (shared with
        # recognize_crop_with_prompt). DashScope rejects crops < 10px with HTTP
        # 400; the default 16px adds a safety margin AND filters crops too
        # small to hold legible characters.
        self._min_crop_side = getattr(settings, "vlm_min_crop_side", 16)

    def recognize_crop(self, image, polygon) -> tuple[str, float]:
        # Tight bbox around the quad (art text on dielines is mostly axis-aligned).
        bbox = self._crop_bbox_or_none(image, polygon)
        if bbox is None:
            return "", 0.0
        x1, y1, x2, y2 = bbox

        crop = image[y1:y2, x1:x2]
        b64 = _to_b64_jpeg(crop)

        text, _conf = self.ask_image(
            b64, _PROMPT, max_tokens=128, json_mode=False
        )
        text = text.strip()
        if not text or text.upper() == "EMPTY":
            return "", 0.0
        # The self-rating prompt appends "||<score>"; parse it into a real
        # confidence. Falls back to _FALLBACK_CONF (0.8) if the model didn't
        # follow the format — same behavior as before this change.
        text, score = _parse_self_rated(text)
        text = re.sub(r"^['\"]|['\"]$", "", text)
        return text, score

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

        Confidence is a flat 0.8 placeholder: Qwen-VL gives no native score, so
        this method returns a constant. Callers that need a real score use the
        self-rating prompt (``_PROMPT``) and :func:`_parse_self_rated` via
        ``recognize_crop`` / ``recognize_crop_with_prompt``, which parse a
        ``text||score`` response. Every current ``ask_image`` caller discards
        the returned confidence (``_conf``), so the placeholder is harmless.
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
