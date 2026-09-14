"""Qwen-VL (OpenAI-compatible) provider for art-text fallback.

DashScope exposes an OpenAI-compatible endpoint; we send the cropped region as
a base64 image with a short, strict prompt asking for the literal text only.
This avoids the "VLM emits coordinates" problem entirely — we just want chars.

The VLM's job is text recognition ONLY. It does not rate confidence: empirically
its self-rated scores were unreliable (art-text over-modesty, format drift into
JSON). The pipeline keeps the original PaddleOCR confidence for any box the VLM
re-reads; "success" here is simply "the VLM returned non-empty text".
"""
from __future__ import annotations

import base64
import io
import json
import logging
import re

from ..config import Settings
from .base import VLMProvider

logger = logging.getLogger(__name__)

_PROMPT = (
    "Read out the text visible in this image, exactly as written. "
    "Return the text ONLY when you are confident you can read EVERY character "
    "correctly — i.e. the text is clear and legible, not blurred/truncated/"
    "obscured. If ANY part is unclear, partially hidden, or you would have to "
    "guess, do NOT return a partial or guessed result: output the single word "
    "EMPTY instead. Accuracy over coverage — an honest EMPTY is better than a "
    "wrong guess. "
    "When you do return text, output ONLY the recognized text itself — no "
    "quotes, no commentary, no JSON, no code fences. "
    "Subscripts/superscripts (chemical formulas, units) must be written as "
    "plain characters inline (C60, H2O, m2) — never LaTeX/math markup "
    "(no $...$, no \\text, no _{} or ^{})."
)

# LaTeX math markup that VLMs habitually emit for sub/superscripted text: the
# chemistry notation C₆₀ comes back as ``$\text{C}_{60}$`` instead of "C60".
# Packaging copy is plain text, so we normalize it back to literal characters.
# Any ``\command{arg}`` keeps its arg; braced / single-char ``_``/``^`` keep
# their characters inline; leftover bare commands and math delimiters drop.

# \text{C} / \mathrm{H} / \mathbf{x} ... → inner content (one nesting level
# per pass; applied in a loop).
_LATEX_CMD_ARG_RE = re.compile(r"\\[A-Za-z]+\s*\{([^{}]*)\}")
# _{60} / ^{2} → inner content.
_LATEX_SUBSUP_BRACE_RE = re.compile(r"[_^]\s*\{([^{}]*)\}")
# _2 / ^n → the character (brace-less single-char form).
_LATEX_SUBSUP_SINGLE_RE = re.compile(r"[_^]\s*([A-Za-z0-9])")
# Escaped specials \% \& \_ \{ \} \# → the literal character.
_LATEX_ESCAPE_RE = re.compile(r"\\([%&_{}#])")
# Leftover bare commands (\cdot, \left, \,) and math-mode delimiters \( \) \[
# \] — dropped. Runs only when the string already looks like LaTeX, so a
# backslash in ordinary copy is never touched.
_LATEX_CMD_BARE_RE = re.compile(r"\\(?:[A-Za-z]+|.)")


def _looks_like_latex(s: str) -> bool:
    r"""Heuristic gate for LaTeX normalization.

    Fires on: formula markup between PAIRED ``$...$`` (``$H_2O$``), a braced
    sub/superscript anywhere (``C_{60}``, ``10^{6}``), or a ``\command`` token
    / ``\(`` ``\[`` math-mode opener (``$\text{C}_{60}$``). Deliberately does
    NOT fire on a lone ``$`` next to a bare ``_`` — prices ("$9.9") and
    snake_case must pass through untouched; normalizing those would corrupt
    legitimate copy.
    """
    if re.search(r"\$[^$]+[_^][^$]*\$", s):  # markup between paired $...$
        return True
    if re.search(r"[_^]\s*\{", s):  # braced sub/superscript: C_{60}, 10^{6}
        return True
    return bool(re.search(r"\\[A-Za-z]+|\\\(|\\\[", s))


def _strip_latex(s: str) -> str:
    r"""Normalize LaTeX math markup to plain characters.

    ``$\text{C}_{60}$`` → ``C60``, ``$\mathrm{H_2O}$`` → ``H2O``,
    ``$V_{max}$`` → ``Vmax``. Superscript semantics are intentionally
    flattened (``10^{6}`` → ``106``) — packaging copy needs the characters,
    not the typesetting. Applied ONLY when the string looks like LaTeX (see
    :func:`_looks_like_latex`) so ordinary text with ``$`` passes through
    unchanged.
    """
    if not _looks_like_latex(s):
        return s
    out = s
    for _ in range(3):  # unwind nested \cmd{...\cmd{...}} one level per pass
        replaced = _LATEX_CMD_ARG_RE.sub(r"\1", out)
        if replaced == out:
            break
        out = replaced
    out = _LATEX_SUBSUP_BRACE_RE.sub(r"\1", out)
    out = _LATEX_SUBSUP_SINGLE_RE.sub(r"\1", out)
    out = _LATEX_ESCAPE_RE.sub(r"\1", out)
    out = _LATEX_CMD_BARE_RE.sub("", out)
    out = out.replace("$", "")
    return re.sub(r"\s{2,}", " ", out).strip()


def _clean_vlm_text(raw: str) -> str | None:
    """Normalize a VLM recognition response into clean text, or None on failure.

    The prompt asks for the literal text only (no score, no JSON). The model
    usually complies, but it occasionally goes off-script and emits a JSON
    object — sometimes wrapped in a ```json fence — e.g.::

        ```json
        {"text": "能量宇宙 擎天柱", "score": 0.0}
        ```

    Returning that verbatim as ``text`` poisons the OCR result (a downstream
    caller sees a box whose text is the raw JSON string). We refuse to guess
    the model's intent in that case and return ``None`` so the caller treats it
    as a failed read and keeps the original PaddleOCR result.

    Returns:
        The cleaned text (leading/trailing quotes/whitespace stripped, LaTeX
        math markup normalized to plain characters), or ``None`` when the
        response is empty, the literal ``EMPTY`` sentinel, or a JSON/fenced
        block (model abandoned the plain-text format).
    """
    s = (raw or "").strip()
    if not s or s.upper() == "EMPTY":
        return None
    # Markdown code fence (``` or ```json ... ```): the model wrapped its
    # output in a fence — a clear sign it abandoned the requested plain format.
    if s.startswith("```"):
        logger.info(
            "vlm text rejected: fenced/JSON output instead of plain text; "
            "raw=%r", raw[:80],
        )
        return None
    # A JSON object with text/score keys: the model emitted structured output.
    # Tolerate leading/trailing whitespace; other JSON-like text (e.g. a real
    # OCR of '{"price": 9.9}') without those keys is kept as literal text.
    if s.startswith("{"):
        try:
            obj = json.loads(s)
        except (ValueError, TypeError):
            obj = None
        if isinstance(obj, dict) and ("text" in obj or "score" in obj):
            logger.info(
                "vlm text rejected: JSON object with text/score key; raw=%r",
                raw[:80],
            )
            return None
    # Strip a leading/trailing quote the model sometimes adds, then normalize
    # LaTeX math markup ($\text{C}_{60}$ → C60) — the model's habitual rendering
    # of sub/superscripted formulas, which must not leak into OCR copy verbatim.
    return _strip_latex(re.sub(r"^['\"]|['\"]$", "", s))



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

        raw, _conf = self.ask_image(
            b64, _PROMPT, max_tokens=128, json_mode=False
        )
        # The VLM does text recognition only — no self-rated score. _clean_vlm_text
        # returns None for empty / EMPTY / JSON-fenced garbage, else the text.
        text = _clean_vlm_text(raw)
        if text is None:
            logger.info(
                "recognize_crop: bbox=[%d,%d,%d,%d] %dx%d -> EMPTY/garbage "
                "(raw=%r)",
                x1, y1, x2, y2, x2 - x1, y2 - y1, (raw or "")[:80],
            )
            return "", 0.0
        # Confidence is a placeholder: the pipeline ignores it and keeps the
        # original PaddleOCR score for this box. 1.0 just signals "read OK".
        logger.info(
            "recognize_crop: bbox=[%d,%d,%d,%d] %dx%d -> text=%r (raw=%r)",
            x1, y1, x2, y2, x2 - x1, y2 - y1, text[:80], (raw or "")[:80],
        )
        return text, 1.0

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

        Confidence is a flat placeholder: Qwen-VL gives no native score, so this
        method always returns 0.8, which every caller discards. The art-text /
        circular fallback paths no longer ask the VLM for a self-rated score
        (it was unreliable) — they keep the original PaddleOCR confidence for
        any re-read box and treat "non-empty VLM text" as success. Only the
        VLM grounding OCR path (``_OCR_PROMPT``) still reads a per-item
        ``confidence`` field from its JSON (parsed in
        ``vlm_ocr._norm_confidence``), and that is independent of this method.
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
        import time
        _t = time.perf_counter()
        resp = self._client.chat.completions.create(**kwargs)
        _dt = time.perf_counter() - _t
        msg = resp.choices[0].message
        text = (getattr(msg, "content", None) or "").strip()
        logger.info(
            "ask_image: model=%s think=%s json=%s tokens=%d len=%d %.2fs",
            kwargs["model"], want_thinking, use_json,
            getattr(getattr(resp, "usage", None), "total_tokens", 0) or 0,
            len(text), _dt,
        )
        return text, 0.8


def _to_b64_jpeg(arr) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="JPEG", quality=92)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"
