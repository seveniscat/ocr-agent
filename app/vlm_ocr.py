"""VLM OCR engine — Qwen-VL grounding OCR as an alternative to PaddleOCR.

This is a peer of :mod:`app.pipeline`'s PaddleOCR path: it takes the whole
(autocropped) image, tiles it so each tile fits the VLM's effective resolution
(~2K), asks the VLM to ground every text/art_text/qr/barcode with a normalized
``[x1,y1,x2,y2]`` box, then maps those back to pixel-space quads in the global
image. The output is the same :class:`Item` list PaddleOCR produces, so the
pipeline's dedupe / paragraph-merge / annotate stages run unchanged.

Why tile instead of one whole-image call
----------------------------------------
Die-lines run 1000–10000px but Qwen-VL resolves detail to ~2K. A single call on
a 10000px image (whether base64-inlined or URL'd) either blows the request size
or downscales past readability — small copy text vanishes and bboxes go coarse.
Tiling keeps each VLM call small (~hundreds of KB base64) and each box tight, at
the cost of more calls (bounded by ``plan_grid``).

Resilience
----------
Like the understanding layer, a VLM that returns garbage must never break the
request. Per-tile calls are wrapped; a tile that fails is skipped (logged) and
the rest proceed. JSON parsing is tolerant (markdown fences, prose preamble,
``<box>`` tags, swapped corners, values >1). ``run_vlm_ocr`` never raises on
model/parse errors — it returns whatever it salvaged.

Code channel
------------
Codes (qr/barcode) come from TWO sources here: the VLM (visual grounding — finds
artistic/rotated codes pyzbar misses, but can be loose on the payload) and
pyzbar (reliable decoding, when the lib is present). Both are emitted; the
pipeline's ``dedupe_items`` folds overlaps by geometry + text. So a VLM code
that pyzbar also decoded collapses to one Item, and a VLM-only code survives
with ``source="vlm"``.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from .schemas import Item, ItemType
from .tiling import (
    crop_tile,
    offset_polygon,
    plan_grid,
    polygon_to_bbox,
    tile_specs,
)
from .understanding import _encode_for_vlm, _extract_json
from .vlm.base import VLMProvider

if TYPE_CHECKING:
    import numpy as np

    from .config import Settings

logger = logging.getLogger(__name__)

# Prompt: ask the VLM to ground EVERY readable element as normalized [x1,y1,x2,y2].
# Output schema is fixed (``items`` of {type, text/content, bbox}). We accept
# ``text`` for text/art_text and ``content`` for qr/barcode payloads to mirror
# the Item schema exactly; the parser tolerates either key on either type.
_OCR_PROMPT = """你是一个包装图像 OCR 与定位专家。请仔细观察这张图片，找出图中【所有】可识别的元素，并为每个元素给出精确的边界框。

需要识别并框出的元素类型：
- text: 普通文字（正文、说明、配料表、净含量等印刷文字）
- art_text: 艺术字 / 变形字 / logo 文字 / 手写体 / 特殊字体
- barcode: 条形码（含 EAN-13、69码等一维条码）
- qr: 二维码（QR 码等二维条码）

输出严格的 JSON，格式如下：
{
  "items": [
    {
      "type": "text | art_text | qr | barcode",
      "text": "识别出的文字内容（text/art_text 用此字段）",
      "content": "条码/二维码解码内容（qr/barcode 用此字段；无法解码则省略）",
      "bbox": [x1, y1, x2, y2]
    }
  ]
}

要求：
- bbox 为归一化坐标，取值 0.0–1.0，[x1,y1] 是元素左上角，[x2,y2] 是右下角（左上角为原点，x 向右 y 向下）。
- 必须框出【所有】可见的文字、艺术字、条码、二维码，不要遗漏，宁可多框也不要漏框。
- 同一行文字应作为一个 text 框（行级），不要拆成单字。
- text/art_text 用 "text" 字段；qr/barcode 用 "content" 字段（若无法解码可不填或留空）。
- 坐标尽量贴合元素边界，不要框得过大。
- 只输出 JSON 对象本身，不要任何解释文字、不要 markdown 代码围栏。"""


def run_vlm_ocr(
    img: "np.ndarray",
    vlm: VLMProvider,
    settings: "Settings",
    image_url: str | None = None,
) -> list[Item]:
    """Run VLM grounding OCR. Returns global-coord Items.

    Two image-input modes:

    - **URL mode** (``image_url`` given): pass the public image URL straight to
      the VLM as ``image_url`` (the OpenAI-compatible endpoint fetches it
      server-side). No tiling, no base64 — matches the proven calling pattern
      and avoids ballooning the request on large images. The VLM returns
      normalized [x1,y1,x2,y2] boxes for the WHOLE image; we scale them to the
      image's pixel dimensions (read from ``img``).
    - **bytes mode** (no URL): tile the image and send each tile base64-inlined.
      Used when the caller only has bytes (e.g. a multipart file upload). Each
      tile ≤ ``vlm_ocr_max_side`` so requests stay small.

    Codes are also run through pyzbar (whole image in URL mode, per-tile in
    bytes mode) when the lib is available — the VLM finds artistic codes,
    pyzbar decodes them reliably. The pipeline's dedupe stage collapses overlap.

    Never raises on model/parse failure: the VLM call is wrapped; on error we
    log + return whatever was salvaged (possibly empty).
    """
    h, w = img.shape[:2]
    model = settings.vlm_ocr_model or settings.vlm_model
    confidence = settings.vlm_ocr_confidence
    codes = _get_codes_or_none(settings)
    all_items: list[Item] = []

    # =====================================================================
    # URL mode: whole-image VLM call, no tiling, no base64.
    # =====================================================================
    if image_url:
        try:
            raw_items = _detect_from_url(
                vlm, image_url, model=model, img_w=w, img_h=h,
            )
        except Exception as exc:  # noqa: BLE001 — best-effort
            logger.warning("vlm_ocr: whole-image VLM call failed: %s", exc)
            raw_items = []

        for itype, payload, norm in raw_items:
            # norm is canonical [x1,y1,x2,y2] in [0,1] → scale to full image px.
            x1, y1 = norm[0] * w, norm[1] * h
            x2, y2 = norm[2] * w, norm[3] * h
            quad = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            all_items.append(
                _build_item(itype, payload, quad, confidence, None)
            )

        # pyzbar on the whole image (reliable decoding, complements VLM boxes).
        if codes is not None:
            try:
                for det in codes.detect(img):
                    all_items.append(
                        Item(
                            id="tmp",
                            type=det.type,
                            content=det.content,
                            polygon=det.polygon,
                            bbox=polygon_to_bbox(det.polygon),
                            confidence=det.confidence,
                            source="pyzbar",
                        )
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning("vlm_ocr: code detection failed: %s", exc)

        logger.info(
            "vlm_ocr[url]: %dx%d items=%d model=%s url=%.80s",
            w, h, len(all_items), model, image_url,
        )
        return all_items

    # =====================================================================
    # Bytes mode: tile + base64 per tile (fallback for multipart uploads).
    # =====================================================================
    target_size = (
        max(w, h)
        if max(w, h) <= settings.vlm_ocr_max_side
        else settings.vlm_ocr_max_side
    )
    grid = plan_grid(
        w, h, target_size=target_size, overlap=settings.tile_overlap,
    )
    specs = tile_specs(grid)
    max_side = settings.vlm_ocr_max_side

    for spec in specs:
        tile = crop_tile(img, spec)
        th, tw = tile.shape[:2]
        try:
            raw_items = _detect_tile(vlm, tile, model=model, max_side=max_side)
        except Exception as exc:  # noqa: BLE001 — best-effort per tile
            logger.warning(
                "vlm_ocr: tile %d VLM call failed, skipping: %s", spec.index, exc
            )
            raw_items = []

        for itype, payload, norm in raw_items:
            x1, y1 = norm[0] * tw, norm[1] * th
            x2, y2 = norm[2] * tw, norm[3] * th
            quad = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
            global_poly = offset_polygon(quad, spec.x0, spec.y0)
            all_items.append(
                _build_item(itype, payload, global_poly, confidence, spec.index)
            )

        if codes is not None:
            try:
                for det in codes.detect(tile):
                    global_poly = offset_polygon(
                        det.polygon, spec.x0, spec.y0
                    )
                    all_items.append(
                        Item(
                            id="tmp",
                            type=det.type,
                            content=det.content,
                            polygon=global_poly,
                            bbox=polygon_to_bbox(global_poly),
                            confidence=det.confidence,
                            source="pyzbar",
                            tile_index=spec.index,
                        )
                    )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "vlm_ocr: code detection failed on tile %d: %s",
                    spec.index, exc,
                )

    logger.info(
        "vlm_ocr[bytes]: %dx%d tiles=%d items=%d model=%s",
        w, h, grid.count, len(all_items), model,
    )
    return all_items


# ---------------------------------------------------------------------------
# Per-tile detection
# ---------------------------------------------------------------------------


def _detect_tile(
    vlm: VLMProvider,
    tile: "np.ndarray",
    *,
    model: str,
    max_side: int,
) -> list[tuple[str, str, list[float]]]:
    """Ground all elements in one tile via the VLM.

    Returns a list of ``(type, payload, norm_bbox)`` where ``payload`` is the
    text (for text/art_text) or decoded content (for qr/barcode), and
    ``norm_bbox`` is ``[x1,y1,x2,y2]`` in 0–1 fractions of the tile.

    Raises whatever the VLM call raises — the caller wraps that per-tile.
    """
    th, tw = tile.shape[:2]
    data_url = _encode_for_vlm(tile, max_side=max_side)
    raw, _conf = vlm.ask_image(
        data_url, _OCR_PROMPT, max_tokens=8192, json_mode=True,
        model_override=model or None,
    )
    return _parse_ocr_items(raw, img_w=tw, img_h=th)


def _detect_from_url(
    vlm: VLMProvider,
    image_url: str,
    *,
    model: str,
    img_w: int = 0,
    img_h: int = 0,
) -> list[tuple[str, str, list[float]]]:
    """Ground all elements in a whole image via a public URL.

    Passes ``image_url`` straight to the VLM (the OpenAI-compatible endpoint
    fetches it server-side) — no tiling, no base64. Returns ``(type, payload,
    norm_bbox)`` with ``norm_bbox`` in 0–1 fractions of the whole image. The
    caller scales by the image's pixel dimensions.

    Raises whatever the VLM call raises — the caller wraps it.
    """
    # ask_image accepts a full data URL OR a public http(s) URL in the
    # ``image_url`` slot (DashScope/Qwen-VL fetches it server-side). We pass the
    # raw URL through unchanged. max_tokens is generous (8192) because a dense
    # packaging image can yield 100+ items whose JSON exceeds 4K tokens.
    raw, _conf = vlm.ask_image(
        image_url, _OCR_PROMPT, max_tokens=8192, json_mode=True,
        model_override=model or None,
    )
    return _parse_ocr_items(raw, img_w=img_w, img_h=img_h)


def _parse_ocr_items(
    raw: str,
    *,
    img_w: int = 0,
    img_h: int = 0,
) -> list[tuple[str, str, list[float]]]:
    """Parse VLM output into ``(type, payload, norm_bbox)`` triples.

    Tolerant of: markdown fences / prose preamble, truncated JSON (recovers
    complete items before the cut), ``bbox`` as pixel coords vs normalized
    (auto-detects: when values exceed ~1.5 they're treated as pixels and
    divided by ``img_w``/``img_h``), swapped corners, and values slightly
    outside [0,1] (clamped). Unknown types / bbox-less entries are dropped.
    """
    items = _extract_items(raw)
    if not items:
        logger.warning("vlm_ocr: no items in VLM output: %.200s", raw)
        return []

    # Detect whether bboxes are pixel coords or normalized. If ANY item has a
    # value > 1.5, the model returned absolute pixels (qwen3.7-plus often does
    # this despite the prompt asking for 0–1). Normalize using the image size.
    pixel_mode = False
    for e in items:
        b = e.get("bbox") if isinstance(e, dict) else None
        if isinstance(b, list) and len(b) == 4:
            try:
                if max(float(x) for x in b) > 1.5:
                    pixel_mode = True
                    break
            except (TypeError, ValueError):
                continue

    out: list[tuple[str, str, list[float]]] = []
    for e in items:
        if not isinstance(e, dict):
            continue
        itype = _norm_type(e.get("type"))
        if itype is None:
            continue
        norm = _norm_bbox(e, pixel_mode=pixel_mode, img_w=img_w, img_h=img_h)
        if norm is None:
            continue
        # payload: text for text/art_text, content for qr/barcode — but accept
        # whichever key the model actually provided.
        payload = ""
        if itype in ("qr", "barcode"):
            payload = str(e.get("content") or e.get("text") or "").strip()
        else:
            payload = str(e.get("text") or e.get("content") or "").strip()
        out.append((itype, payload, norm))
    return out


def _extract_items(raw: str) -> list[dict]:
    """Extract the ``items`` list from VLM JSON, tolerating truncation.

    Tries, in order: (1) clean full-JSON parse, (2) regex to the last complete
    ``}`` that still yields valid JSON, (3) per-item regex recovery — grab each
    complete ``{ ... }`` object inside the ``items`` array even when the overall
    JSON is cut off mid-object. This last path is what saves a dense-image
    response whose JSON was truncated by the token limit.
    """
    if not raw or not raw.strip():
        return []
    obj = _extract_json(raw)
    if obj is not None and isinstance(obj.get("items"), list):
        return [e for e in obj["items"] if isinstance(e, dict)]

    # Truncation recovery: pull out each complete {...} object that looks like
    # an item (has a "type" or "bbox" key). The outer array may be unclosed,
    # but individual objects up to the cut are still well-formed JSON.
    import json as _json
    import re as _re

    item_objs: list[dict] = []
    for m in _re.finditer(r"\{[^{}]*\}", raw):
        chunk = m.group(0)
        try:
            e = _json.loads(chunk)
        except _json.JSONDecodeError:
            continue
        if isinstance(e, dict) and ("type" in e or "bbox" in e):
            item_objs.append(e)
    if item_objs:
        logger.info(
            "vlm_ocr: recovered %d items from truncated JSON", len(item_objs)
        )
    return item_objs


def _norm_type(v) -> str | None:
    """Coerce the model's type string to one of our ItemType values."""
    if not isinstance(v, str):
        return None
    t = v.strip().lower()
    if t in ("text", "art_text", "qr", "barcode"):
        return t
    # Common synonyms / fuzzy forms.
    if t in ("art", "arttext", "logo_text", "stylized"):
        return "art_text"
    if t in ("qrcode", "qrc", "qrcode_code"):
        return "qr"
    if t in ("bar", "bar_code", "ean", "ean13", "code128", "条码", "一维码"):
        return "barcode"
    if t in ("txt", "string", "字", "文字"):
        return "text"
    return None


def _norm_bbox(
    e: dict,
    *,
    pixel_mode: bool = False,
    img_w: int = 0,
    img_h: int = 0,
) -> list[float] | None:
    """Extract a normalized ``[x1,y1,x2,y2]`` from a VLM item dict.

    Accepts ``bbox`` / ``box`` / ``bbox_2d`` keys. When ``pixel_mode`` is True
    the raw values are treated as absolute pixels and divided by ``img_w`` /
    ``img_h`` (x by width, y by height) to normalize — qwen3.7-plus often
    returns pixel coords despite the prompt asking for 0–1. Also strips the
    Qwen ``<box>...</box>`` string form. Returns None when no usable 4-float
    box can be recovered.
    """
    for key in ("bbox", "box", "bbox_2d"):
        v = e.get(key)
        if isinstance(v, list) and len(v) == 4:
            try:
                vals = [float(x) for x in v]
            except (TypeError, ValueError):
                continue
            return _canonicalize_bbox(
                vals, pixel_mode=pixel_mode, img_w=img_w, img_h=img_h
            )
        if isinstance(v, str):
            # "<box>x1,y1,x2,y2</box>" or bare "x1,y1,x2,y2"
            import re

            m = re.findall(r"-?\d+\.?\d*", v)
            if len(m) == 4:
                try:
                    return _canonicalize_bbox(
                        [float(x) for x in m],
                        pixel_mode=pixel_mode, img_w=img_w, img_h=img_h,
                    )
                except (TypeError, ValueError):
                    pass
    return None


def _canonicalize_bbox(
    vals: list[float],
    *,
    pixel_mode: bool = False,
    img_w: int = 0,
    img_h: int = 0,
) -> list[float]:
    """Turn a 4-number list into a clamped, sorted ``[x1,y1,x2,y2]`` in [0,1].

    Treats the 4 values as two corners ``[x1,y1,x2,y2]`` and sorts each axis so
    a swapped-corner output still yields a valid box. In ``pixel_mode`` the
    x-values are divided by ``img_w`` and y-values by ``img_h`` first (clamped
    to a safe range to reject absurd model output).
    """
    a, b, c, d = vals
    if pixel_mode and img_w > 0 and img_h > 0:
        # Normalize pixels → fractions. Clamp inputs to a generous bound so a
        # stray huge value doesn't produce a nonsense fraction.
        a = max(0.0, min(a, img_w)) / img_w
        c = max(0.0, min(c, img_w)) / img_w
        b = max(0.0, min(b, img_h)) / img_h
        d = max(0.0, min(d, img_h)) / img_h
    x1, x2 = sorted([_clamp01(a), _clamp01(c)])
    y1, y2 = sorted([_clamp01(b), _clamp01(d)])
    return [x1, y1, x2, y2]


def _clamp01(v: float) -> float:
    return max(0.0, min(1.0, float(v)))


def _build_item(
    itype: str,
    payload: str,
    global_poly: list[list[float]],
    confidence: float,
    tile_index: int,
) -> Item:
    """Construct an Item from a VLM detection (text vs code payload rules)."""
    if itype in ("qr", "barcode"):
        return Item(
            id="tmp",
            type=itype,  # type: ignore[arg-type]
            content=payload or None,
            polygon=global_poly,
            bbox=polygon_to_bbox(global_poly),
            confidence=confidence,
            source="vlm",
            tile_index=tile_index,
        )
    return Item(
        id="tmp",
        type=itype,  # type: ignore[arg-type]
        text=payload or None,
        polygon=global_poly,
        bbox=polygon_to_bbox(global_poly),
        confidence=confidence,
        source="vlm",
        tile_index=tile_index,
        granularity="line",
    )


# Lazily build the code engine the same way Pipeline does (best-effort: returns
# None when the zbar lib is unavailable, so the code channel just goes quiet
# rather than breaking OCR).
def _get_codes_or_none(settings: "Settings"):  # noqa: ARG001 — kept for symmetry
    from .codes.qrcode import CodeEngine

    engine = CodeEngine()
    if engine.available():
        return engine
    logger.warning(
        "vlm_ocr: code engine unavailable (zbar lib missing); pyzbar channel "
        "disabled. VLM-only code boxes will still be returned."
    )
    return None
