"""Tests for the AI-Native understanding layer.

No network / no real VLM: the provider's ``ask_image`` is monkeypatched to
return canned text. Mirrors the style of test_preprocess.py — pure logic tests.
"""
from __future__ import annotations

import io
import json

import numpy as np
from PIL import Image

from app import understanding as und
from app.config import Settings
from app.vlm.qwen import QwenVLM


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _png_bytes(arr: np.ndarray) -> bytes:
    """Encode an HxWx3 uint8 array as PNG bytes (what the endpoint receives)."""
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _solid(w: int, h: int, rgb=(220, 30, 50)) -> np.ndarray:
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:] = rgb
    return arr


def _run(captured: dict, raw_return: str):
    """Call understand_image with a fake VLM that returns ``raw_return``.

    ``captured`` is filled with the (data_url, prompt, json_mode) the VLM saw,
    so tests can assert on the encoded image / request shape.
    """
    settings = Settings(understand_enabled=True, understand_max_side=1080)

    def fake_ask(self, image_b64_data_url, prompt, *, max_tokens=1024, json_mode=False):
        captured["data_url"] = image_b64_data_url
        captured["prompt"] = prompt
        captured["json_mode"] = json_mode
        return raw_return, 0.8

    # Monkeypatch the method on the class so understand_image's provider call
    # goes through it. Bypass __init__ to avoid needing an API key / client.
    vlm = QwenVLM.__new__(QwenVLM)
    vlm.name = "qwen"
    vlm.ask_image = fake_ask.__get__(vlm, QwenVLM)  # type: ignore[assignment]

    img = _png_bytes(_solid(600, 400))
    return und.understand_image(img, vlm=vlm, settings=settings)


# ---------------------------------------------------------------------------
# parsing the happy path
# ---------------------------------------------------------------------------


def test_understand_image_parses_valid_json():
    raw = json.dumps({
        "category": "食品-饮料",
        "category_confidence": 0.9,
        "panel_count_estimate": 6,
        "style_keywords": ["极简", "高饱和"],
        "dominant_colors": ["#E63946", "#F1FAEE"],
        "key_elements": [
            {"kind": "logo", "description": "品牌 logo", "location": [0.1, 0.1, 0.2, 0.1]},
            {"kind": "nutrition_table", "description": "营养成分表"},
        ],
        "summary": "一款饮料的纸盒展开图",
    })
    captured: dict = {}
    res = _run(captured, raw)

    assert res.category == "食品-饮料"
    assert res.category_confidence == 0.9
    assert res.panel_count_estimate == 6
    assert res.style_keywords == ["极简", "高饱和"]
    assert res.dominant_colors == ["#E63946", "#F1FAEE"]
    assert len(res.key_elements) == 2
    assert res.key_elements[0].kind == "logo"
    assert res.key_elements[0].location == [0.1, 0.1, 0.2, 0.1]
    assert res.key_elements[1].kind == "nutrition_table"
    assert res.key_elements[1].location is None
    assert res.summary == "一款饮料的纸盒展开图"
    assert res.raw_note is None
    assert res.model == "qwen"


def test_understand_image_sends_json_mode_and_prompt():
    captured: dict = {}
    _run(captured, json.dumps({"category": "其他", "summary": "x"}))
    assert captured["json_mode"] is True
    assert "JSON" in captured["prompt"] or "json" in captured["prompt"]
    assert captured["data_url"].startswith("data:image/jpeg;base64,")


# ---------------------------------------------------------------------------
# parsing resilience (never raises)
# ---------------------------------------------------------------------------


def test_understand_image_handles_markdown_fenced_json():
    raw = '```json\n{"category": "日化", "category_confidence": 0.7, "summary": "ok"}\n```'
    captured: dict = {}
    res = _run(captured, raw)
    assert res.category == "日化"
    assert res.category_confidence == 0.7
    assert res.raw_note is None  # parsed fine


def test_understand_image_handles_prose_wrapped_json():
    raw = '好的，这是结果：\n{"category": "药品", "category_confidence": 0.6, "summary": "y"}\n以上。'
    captured: dict = {}
    res = _run(captured, raw)
    assert res.category == "药品"
    assert res.category_confidence == 0.6


def test_understand_image_garbage_falls_back_without_raising():
    captured: dict = {}
    res = _run(captured, "这不是 JSON，完全是胡言乱语")
    assert res.category_confidence == 0.0
    assert res.raw_note == "这不是 JSON，完全是胡言乱语"
    assert res.model == "qwen"


def test_understand_image_empty_output_falls_back():
    captured: dict = {}
    res = _run(captured, "")
    assert res.category_confidence == 0.0


def test_understand_image_coerces_loose_fields():
    # confidence as string, panel count out of range, unknown kind, extra key.
    raw = json.dumps({
        "category": "食品-零食",
        "category_confidence": "0.85",        # string → coerced
        "panel_count_estimate": 99,           # out of range → clamped
        "key_elements": [
            {"kind": "WRONG", "description": "x"},          # unknown kind → other
            {"description": ""},                             # no description → dropped
            {"kind": "qr", "description": "二维码", "location": [0.1, 0.1, 0.1, 0.1]},
        ],
        "unknown_field": "ignored",
        "summary": "s",
    })
    captured: dict = {}
    res = _run(captured, raw)
    assert res.category_confidence == 0.85
    assert res.panel_count_estimate == 20  # clamped to max
    assert len(res.key_elements) == 2      # empty-description one dropped
    assert res.key_elements[0].kind == "other"  # coerced
    assert res.key_elements[1].kind == "qr"


# ---------------------------------------------------------------------------
# large-image downscaling
# ---------------------------------------------------------------------------


def test_understand_image_downscales_large_image():
    captured: dict = {}
    settings = Settings(understand_enabled=True, understand_max_side=1080)

    def fake_ask(self, image_b64_data_url, prompt, *, max_tokens=1024, json_mode=False):
        captured["data_url"] = image_b64_data_url
        return json.dumps({"category": "x", "summary": ""}), 0.8

    vlm = QwenVLM.__new__(QwenVLM)
    vlm.name = "qwen"
    vlm.ask_image = fake_ask.__get__(vlm, QwenVLM)  # type: ignore[assignment]

    # 5000x3000 image → must be downscaled to long edge 1080 before encoding.
    big = _png_bytes(_solid(5000, 3000))
    und.understand_image(big, vlm=vlm, settings=settings)

    # Decode the data URL the VLM received and confirm its long edge <= 1080.
    header, b64 = captured["data_url"].split(",", 1)
    import base64
    arr = np.array(Image.open(io.BytesIO(base64.b64decode(b64))))
    assert max(arr.shape[:2]) <= 1080


def test_understand_image_does_not_upscale_small_image():
    captured: dict = {}
    settings = Settings(understand_enabled=True, understand_max_side=1080)

    def fake_ask(self, image_b64_data_url, prompt, *, max_tokens=1024, json_mode=False):
        captured["data_url"] = image_b64_data_url
        return json.dumps({"category": "x", "summary": ""}), 0.8

    vlm = QwenVLM.__new__(QwenVLM)
    vlm.name = "qwen"
    vlm.ask_image = fake_ask.__get__(vlm, QwenVLM)  # type: ignore[assignment]

    small = _png_bytes(_solid(200, 100))
    und.understand_image(small, vlm=vlm, settings=settings)

    import base64
    _, b64 = captured["data_url"].split(",", 1)
    arr = np.array(Image.open(io.BytesIO(base64.b64decode(b64))))
    assert arr.shape[1] == 200  # width preserved (no upscaling)


# ---------------------------------------------------------------------------
# _encode_for_vlm + VLM failure path
# ---------------------------------------------------------------------------


def test_understand_image_vlm_exception_falls_back():
    settings = Settings(understand_enabled=True, understand_max_side=1080)

    def fake_ask(self, *a, **k):
        raise RuntimeError("network down")

    vlm = QwenVLM.__new__(QwenVLM)
    vlm.name = "qwen"
    vlm.ask_image = fake_ask.__get__(vlm, QwenVLM)  # type: ignore[assignment]

    res = und.understand_image(_png_bytes(_solid(100, 100)), vlm=vlm, settings=settings)
    assert res.category_confidence == 0.0
    assert "network down" in (res.raw_note or "")


def test_encode_for_vlm_returns_jpeg_data_url():
    arr = _solid(300, 200)
    url = und._encode_for_vlm(arr, max_side=1080)
    assert url.startswith("data:image/jpeg;base64,")
    import base64
    img = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    assert max(img.size) <= 1080
    assert img.size == (300, 200)  # not upscaled
