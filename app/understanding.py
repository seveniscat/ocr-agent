"""VLM image understanding layer — "what is this image?".

A standalone branch that does NOT touch the OCR pipeline: it takes the raw
image, downscales to a VLM-friendly size, asks the VLM one structured question,
and returns a typed :class:`UnderstandingResult`. This is the first AI-Native
capability — the VLM graduates from "art-text recognition fallback" to "image
understander".

Large-image handling
--------------------
Dielines run 1000–10000px but VLMs effectively resolve ~2Kpx. For whole-image
*understanding* (level 1) we don't need to read every character of the
ingredients list — we just need to "see what it is" — so a single downscale to
the VLM's sweet spot (~1080px long edge) is sufficient. Per-panel VLM calls
(level 2) are a future extension that reuses :func:`app.panels.split_panels`.

Resilience
----------
A VLM that returns garbage must never break the request. JSON parsing + Pydantic
validation happen inside :func:`_safe_parse`; on any failure we fall back to a
minimal ``UnderstandingResult`` with ``category_confidence=0`` and the raw output
preserved in ``raw_note`` so the UI can show *something*.
"""
from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

from .config import Settings
from .schemas import UnderstandingResult
from .vlm.base import VLMProvider

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)

# Prompt for level-1 whole-image understanding. Output schema is fixed; keep the
# ``kind`` enum here in sync with ``ElementKind`` in schemas.py.
_PROMPT = """你是一位资深的包装图像理解专家。请观察这张图片(可能是包装盒展开图/模切图/标签/实物包装照片),输出严格的 JSON,字段如下:

{
  "category": "包装品类,如 '食品-饮料' / '食品-零食' / '日化' / '化妆品' / '药品' / '电子' / '其他'",
  "category_confidence": 0.0到1.0的浮点数,你对品类判断的把握,
  "panel_count_estimate": 整数,你估算的画面/面板数量(1=单面标签, 4-6=纸盒展开图, 等等),
  "style_keywords": ["风格关键词,最多5个,如 '极简','手绘','高饱和','卡通','国潮'"],
  "dominant_colors": ["主色调十六进制,最多4个,如 '#E63946'"],
  "key_elements": [
    {"kind": "logo|product_image|text_block|barcode|qr|nutrition_table|color_block|other", "description": "简短描述这个元素", "location": [x, y, w, h]}
  ],
  "summary": "一句话概述这张图是什么"
}

其中 location 是归一化坐标 [x, y, w, h],取值 0-1,表示该元素在图中的大致位置(左上角原点)。如果不确定位置可以省略 location 字段。

只输出 JSON 对象本身,不要任何解释文字、不要 markdown 代码围栏。"""


def understand_image(
    image_data: bytes, vlm: VLMProvider, settings: Settings
) -> UnderstandingResult:
    """Whole-image understanding: downscale → encode → ask VLM → parse.

    Parameters
    ----------
    image_data : bytes
        Raw image bytes (PNG/JPEG/...).
    vlm : VLMProvider
        A provider implementing :meth:`ask_image` (e.g. QwenVLM).
    settings : Settings
        Read for ``understand_max_side``.

    Returns
    -------
    UnderstandingResult
        Always returns — never raises on VLM/parse failure (those are caught
        and surfaced via ``category_confidence=0`` + ``raw_note``).
    """
    from .tiling import load_image

    img = load_image(image_data)  # HxWx3 RGB
    data_url = _encode_for_vlm(img, max_side=settings.understand_max_side)

    try:
        raw, _conf = vlm.ask_image(
            data_url, _PROMPT, max_tokens=1024, json_mode=True
        )
    except Exception as exc:  # noqa: BLE001 — VLM is best-effort
        logger.warning("understand: VLM call failed: %s", exc)
        return _fallback("VLM 调用失败", str(exc), model=getattr(vlm, "name", ""))

    return _safe_parse(raw, model=getattr(vlm, "name", ""))


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------


def _encode_for_vlm(img: "np.ndarray", *, max_side: int) -> str:
    """Downscale ``img`` so its long edge <= ``max_side`` and JPEG-encode to a
    ``data:image/jpeg;base64,...`` URL.

    Upscaling is never done — a tiny image stays tiny. LANCZOS for clean
    downscaling of line art / text. Returns the data URL the OpenAI-compatible
    ``image_url`` slot expects.
    """
    import base64
    import io

    from PIL import Image

    h, w = img.shape[:2]
    scale = max_side / max(w, h)
    if scale < 1.0:
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        pil = Image.fromarray(img).resize((new_w, new_h), Image.LANCZOS)
    else:
        pil = Image.fromarray(img)

    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=90)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


# ---------------------------------------------------------------------------
# Parsing — the resilience boundary
# ---------------------------------------------------------------------------


def _safe_parse(raw: str, *, model: str) -> UnderstandingResult:
    """Parse VLM output into UnderstandingResult, never raising.

    Tries (1) direct JSON parse, (2) extract the first ``{...}`` block (handles
    models that wrap output in markdown fences or prose), then validates with
    Pydantic. Any failure → fallback result with ``category_confidence=0`` and
    the raw text preserved.
    """
    if not raw or not raw.strip():
        return _fallback("VLM 返回空", raw, model=model)

    obj = _extract_json(raw)
    if obj is None:
        logger.warning("understand: no JSON found in VLM output: %.200s", raw)
        return _fallback("无法解析 VLM 输出为 JSON", raw, model=model)

    # Coerce / sanitize fields before validation so a slightly-off VLM output
    # still parses (e.g. confidence as a string, panel count out of range).
    obj = _coerce(obj)
    try:
        return UnderstandingResult(model=model, **obj)
    except Exception as exc:  # noqa: BLE001 — Pydantic ValidationError etc.
        logger.warning("understand: validation failed (%s): %.200s", exc, raw)
        return _fallback("JSON 字段校验失败", raw, model=model)


def _extract_json(raw: str) -> dict | None:
    """Return the parsed dict, or None if no JSON object could be extracted."""
    raw = raw.strip()
    # Fast path: clean JSON object.
    try:
        obj = json.loads(raw)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass

    # Slow path: strip markdown fences / leading prose, grab the outermost {...}.
    # DOTALL so '.' crosses newlines (the object spans many lines).
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _coerce(obj: dict) -> dict:
    """Best-effort field normalization so a slightly-loose VLM output validates.

    Drops unknown keys (Pydantic would reject them by default), clamps ranges,
    coerces obvious type mismatches. Never raises — a key we can't coerce is
    dropped so validation can report a clean error or proceed.
    """
    out: dict = {}

    def _str(key: str, default: str = "") -> None:
        v = obj.get(key)
        out[key] = v if isinstance(v, str) else default

    def _clamp_float(key: str, lo: float, hi: float, default: float) -> None:
        v = obj.get(key, default)
        try:
            f = float(v)
        except (TypeError, ValueError):
            f = default
        out[key] = max(lo, min(hi, f))

    def _clamp_int(key: str, lo: int, hi: int, default: int) -> None:
        v = obj.get(key, default)
        try:
            i = int(v)
        except (TypeError, ValueError):
            i = default
        out[key] = max(lo, min(hi, i))

    def _str_list(key: str) -> None:
        v = obj.get(key, [])
        if isinstance(v, list):
            out[key] = [str(x) for x in v]
        elif isinstance(v, str):
            out[key] = [v] if v else []
        else:
            out[key] = []

    _str("category")
    _clamp_float("category_confidence", 0.0, 1.0, 0.0)
    _clamp_int("panel_count_estimate", 1, 20, 1)
    _str("summary")
    _str_list("style_keywords")
    _str_list("dominant_colors")

    # key_elements — keep only dicts with a description; coerce kind to str.
    allowed_kinds = {
        "logo", "product_image", "text_block", "barcode", "qr",
        "nutrition_table", "color_block", "other",
    }
    elems: list[dict] = []
    raw_elems = obj.get("key_elements", [])
    if isinstance(raw_elems, list):
        for e in raw_elems:
            if not isinstance(e, dict):
                continue
            desc = e.get("description")
            if not isinstance(desc, str) or not desc.strip():
                continue
            kind = str(e.get("kind", "other")).strip()
            if kind not in allowed_kinds:
                kind = "other"
            item: dict = {"kind": kind, "description": desc}
            loc = e.get("location")
            if isinstance(loc, list) and len(loc) == 4:
                try:
                    item["location"] = [float(x) for x in loc]
                except (TypeError, ValueError):
                    pass
            elems.append(item)
    out["key_elements"] = elems
    return out


def _fallback(message: str, raw: str, *, model: str) -> UnderstandingResult:
    """Minimal result used when understanding failed — keeps the request alive."""
    return UnderstandingResult(
        category="未知",
        category_confidence=0.0,
        panel_count_estimate=1,
        summary=message,
        raw_note=raw if raw else message,
        model=model,
    )


# ===========================================================================
# VLM-based panel splitting
#
# Why VLM and not geometry: on finished design drafts (artwork filling each
# face) the panel borders are pixel-wise indistinguishable from in-panel
# content edges, so every pure-geometry method tried (LSD, density projection,
# connected components, change-point) failed across the sample set. The VLM
# reads the *semantic* layout ("these are 6 faces of a box unfolded in a 2×3
# grid") which is exactly what geometry can't recover.
#
# Trade-off accepted: we send the image once at a VLM-friendly resolution
# (~1500px long edge — Qwen-VL localizes reliably up to ~2560). Coordinates
# come back as normalized fractions, so precision is bounded by that downscale
# (≈0.07% of the long edge per step — plenty for face-level boxes, not for
# sub-pixel cut lines). The UI can let the user drag to refine.
# ===========================================================================


_PANELS_PROMPT = """这张图是一个长方体包装盒的刀模展开图（6个面平铺排列）。
请识别图中每一个独立的"面"（即盒子的一个面，被折线/切线围成的矩形区域），输出严格的 JSON：

{
  "panels": [
    {"index": 1, "bbox": [x1, y1, x2, y2], "label": "这个面的类型，从下列选一个: front/back/left/right/top/bottom/unknown"}
  ]
}

要求：
- bbox 是归一化坐标，取值 0.0-1.0，表示该面在图中的位置（左上角原点，x向右y向下）。[x1,y1]是面左上角，[x2,y2]是右下角。
- 通常有 5 或 6 个面。只输出主要的面，忽略小的糊口(glue tab)、插舌(tuck flap)、防尘翼等附属结构。
- 面按从左到右、从上到下的顺序编号。
- 只输出 JSON 对象本身，不要任何解释、不要 markdown 代码围栏。"""


def split_panels_vlm(
    image_data: bytes, vlm: VLMProvider, settings: Settings
) -> dict:
    """VLM-based panel splitting.

    Downscale → encode → ask VLM for normalized per-face bboxes → scale back to
    original-image pixels. Always returns a dict (never raises): on VLM/parse
    failure, ``panels`` is empty and ``error`` carries the reason, so the UI
    can show a message instead of crashing.
    """
    from .tiling import load_image

    img = load_image(image_data)  # HxWx3 RGB
    orig_h, orig_w = img.shape[:2]
    data_url = _encode_for_vlm(img, max_side=settings.understand_max_side)

    try:
        raw, _conf = vlm.ask_image(
            data_url, _PANELS_PROMPT, max_tokens=1024, json_mode=True
        )
    except Exception as exc:  # noqa: BLE001 — VLM best-effort
        logger.warning("split_panels_vlm: VLM call failed: %s", exc)
        return {
            "width": orig_w, "height": orig_h, "count": 0,
            "panels": [], "error": f"VLM 调用失败: {exc}",
            "model": getattr(vlm, "name", ""),
        }

    panels = _parse_panel_boxes(raw, orig_w, orig_h)
    return {
        "width": orig_w, "height": orig_h, "count": len(panels),
        "panels": panels, "raw": raw,
        "model": getattr(vlm, "name", ""),
    }


def _parse_panel_boxes(
    raw: str, orig_w: int, orig_h: int
) -> list[dict]:
    """Parse VLM output into a list of panel dicts with pixel-coord bboxes.

    Tolerant: handles markdown fences, leading prose, bboxes as [x1,y1,x2,y2]
    or [x,y,w,h], and clamps coordinates to [0,1] before scaling to pixels.
    Drops any panel whose bbox is missing/invalid.
    """
    obj = _extract_json(raw)
    if obj is None or not isinstance(obj.get("panels"), list):
        logger.warning("split_panels_vlm: no panels in VLM output: %.200s", raw)
        return []

    out: list[dict] = []
    # VLM may or may not include an index; we assign our own in sort order.
    items = obj["panels"]
    parsed: list[tuple[float, float, float, float, str]] = []
    for e in items:
        if not isinstance(e, dict):
            continue
        bbox = e.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            continue
        try:
            xs = [float(bbox[0]), float(bbox[2])]
            ys = [float(bbox[1]), float(bbox[3])]
        except (TypeError, ValueError):
            continue
        # Clamp to [0,1] and normalize min/max (VLMs sometimes swap corners).
        x1, x2 = sorted(max(0.0, min(1.0, v)) for v in xs)
        y1, y2 = sorted(max(0.0, min(1.0, v)) for v in ys)
        if x2 - x1 < 0.02 or y2 - y1 < 0.02:
            continue  # degenerate box
        label = str(e.get("label", "unknown")).strip().lower() or "unknown"
        parsed.append((x1, y1, x2, y2, label))

    # Sort top-to-bottom then left-to-right, assign 1-based index.
    parsed.sort(key=lambda t: (round(t[1] * 20), t[0]))
    for i, (x1, y1, x2, y2, label) in enumerate(parsed):
        px1, py1 = int(round(x1 * orig_w)), int(round(y1 * orig_h))
        px2, py2 = int(round(x2 * orig_w)), int(round(y2 * orig_h))
        out.append({
            "index": i + 1,
            "bbox": [px1, py1, px2, py2],
            "width": px2 - px1,
            "height": py2 - py1,
            "label": label,
        })
    return out
