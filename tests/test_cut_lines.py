"""Tests for VLM cut-line detection (POST /panels/vlm → detect_cut_lines).

No network / no real VLM: ask_image is stubbed to return canned JSON. Mirrors
the style of test_understanding.py — pure logic tests.
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
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _solid(w: int, h: int, rgb=(220, 30, 50)) -> np.ndarray:
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    arr[:] = rgb
    return arr


def _run(captured: dict, raw_return: str, *, max_side=None, img=(600, 400)):
    """Call detect_cut_lines with a fake VLM returning ``raw_return``.

    ``captured`` is filled with the data URL the VLM saw (to assert downscaled
    size). ``img`` is (w, h) of the synthetic source image.
    """
    settings = Settings()

    def fake_ask(self, image_b64_data_url, prompt, *, max_tokens=512, json_mode=False,
                 enable_thinking=None, model_override=None):
        captured["data_url"] = image_b64_data_url
        captured["prompt"] = prompt
        return raw_return, 0.85

    vlm = QwenVLM.__new__(QwenVLM)  # bypass __init__ (no key needed)
    vlm.name = "qwen"
    vlm.ask_image = fake_ask.__get__(vlm, QwenVLM)  # type: ignore[assignment]

    w, h = img
    data = _png_bytes(_solid(w, h))
    return und.detect_cut_lines(data, vlm=vlm, settings=settings, max_side=max_side)


# ---------------------------------------------------------------------------
# _parse_cut_lines
# ---------------------------------------------------------------------------


def test_parse_cut_lines_normal():
    raw = json.dumps({"h_lines": [0.5], "v_lines": [0.25, 0.75]})
    h, v = und._parse_cut_lines(raw)
    assert h == [0.5]
    assert v == [0.25, 0.75]


def test_parse_cut_lines_drops_flush_with_border():
    # 0.0/1.0 and near-border (<2% / >98%) are dropped to avoid degenerate panels.
    raw = json.dumps({"h_lines": [0.0, 0.5, 1.0, 0.01, 0.99], "v_lines": []})
    h, v = und._parse_cut_lines(raw)
    assert h == [0.5]  # only the interior line survives
    assert v == []


def test_parse_cut_lines_clamps_out_of_range():
    raw = json.dumps({"h_lines": [-0.2, 0.5, 1.5], "v_lines": []})
    h, v = und._parse_cut_lines(raw)
    # -0.2 → 0.0 (dropped as flush), 1.5 → 1.0 (dropped), 0.5 kept
    assert h == [0.5]


def test_parse_cut_lines_dedups_near_equal():
    raw = json.dumps({"h_lines": [0.50, 0.503, 0.51], "v_lines": []})  # within 0.5%
    h, v = und._parse_cut_lines(raw)
    # 0.5 and 0.503 collapse (diff 0.003 ≤ 0.005); 0.51 is 0.007 away → kept
    assert len(h) == 2
    assert h[0] == 0.5


def test_parse_cut_lines_tolerates_markdown_fence():
    raw = '```json\n{"h_lines": [0.3, 0.6], "v_lines": [0.4]}\n```'
    h, v = und._parse_cut_lines(raw)
    assert h == [0.3, 0.6]
    assert v == [0.4]


def test_parse_cut_lines_garbage_returns_empty():
    assert und._parse_cut_lines("not json") == ([], [])
    assert und._parse_cut_lines('{"h_lines": "not a list"}') == ([], [])
    assert und._parse_cut_lines('{"foo": 1}') == ([], [])


def test_parse_cut_lines_coerces_numeric_strings():
    raw = json.dumps({"h_lines": ["0.5", "0.7"], "v_lines": ["0.3"]})
    h, v = und._parse_cut_lines(raw)
    assert h == [0.5, 0.7]
    assert v == [0.3]


# ---------------------------------------------------------------------------
# detect_cut_lines (scales normalized → pixel coords)
# ---------------------------------------------------------------------------


def test_detect_cut_lines_scales_to_pixels():
    # 600x400 image, h_line at 0.5 → y=200, v_lines at 0.25/0.75 → x=150/450
    raw = json.dumps({"h_lines": [0.5], "v_lines": [0.25, 0.75]})
    captured: dict = {}
    res = _run(captured, raw, img=(600, 400))

    assert res["width"] == 600
    assert res["height"] == 400
    assert res["count"] == 3
    h_pos = [l["pos"] for l in res["lines"] if l["orientation"] == "h"]
    v_pos = sorted(l["pos"] for l in res["lines"] if l["orientation"] == "v")
    assert h_pos == [200]            # 0.5 * 400
    assert v_pos == [150, 450]       # 0.25/0.75 * 600
    assert res.get("error") is None


def test_detect_cut_lines_empty_on_no_lines():
    captured: dict = {}
    res = _run(captured, '{"h_lines": [], "v_lines": []}')
    assert res["count"] == 0
    assert res["lines"] == []
    assert res.get("error") is None


def test_detect_cut_lines_falls_back_on_vlm_error():
    settings = Settings()

    def fake_ask(self, *a, **k):
        raise RuntimeError("VLM 挂了")

    vlm = QwenVLM.__new__(QwenVLM)
    vlm.name = "qwen"
    vlm.ask_image = fake_ask.__get__(vlm, QwenVLM)  # type: ignore[assignment]

    res = und.detect_cut_lines(_png_bytes(_solid(100, 100)), vlm=vlm, settings=settings)
    assert res["count"] == 0
    assert "VLM 挂了" in res["error"]
    assert res["lines"] == []


def test_detect_cut_lines_uses_max_side_for_downscale():
    captured: dict = {}
    # 5000x3000 source; force max_side=256 → VLM image long edge ≤ 256
    _run(captured, '{"h_lines":[0.5]}', max_side=256, img=(5000, 3000))
    import base64
    _, b64 = captured["data_url"].split(",", 1)
    arr = np.array(Image.open(io.BytesIO(base64.b64decode(b64))))
    assert max(arr.shape[:2]) <= 256


def test_detect_cut_lines_uses_settings_default_when_no_override():
    captured: dict = {}
    settings = Settings()
    # Default panels_vlm_max_side is 512; 2000x1000 → long edge should be 512
    _run(captured, '{"h_lines":[0.5]}', img=(2000, 1000))
    import base64
    _, b64 = captured["data_url"].split(",", 1)
    arr = np.array(Image.open(io.BytesIO(base64.b64decode(b64))))
    assert max(arr.shape[:2]) <= settings.panels_vlm_max_side


# ---------------------------------------------------------------------------
# endpoint integration
# ---------------------------------------------------------------------------


def test_panels_vlm_endpoint_returns_lines(monkeypatch):
    """The endpoint must return a `lines` array (the new contract) alongside
    the (now empty) `panels`. Mock detect_cut_lines to avoid the network."""
    from fastapi.testclient import TestClient

    from app import main as main_mod

    def fake_detect(image_data, vlm, settings, max_side=None):
        return {"width": 600, "height": 400, "count": 2,
                "lines": [{"pos": 200, "orientation": "h", "confidence": 0.85},
                          {"pos": 300, "orientation": "v", "confidence": 0.85}],
                "model": "test", "error": None}

    monkeypatch.setattr(main_mod, "_settings", lambda: Settings())
    # Stub _get_understand_vlm so the gate passes without a real key.
    monkeypatch.setattr(main_mod, "_get_understand_vlm", lambda: object())
    monkeypatch.setattr(
        "app.understanding.detect_cut_lines", fake_detect
    )

    c = TestClient(main_mod.app)
    import io as _io
    buf = _io.BytesIO()
    Image.new("RGB", (600, 400)).save(buf, format="PNG")
    r = c.post("/panels/vlm", files={"file": ("x.png", buf.getvalue(), "image/png")})
    assert r.status_code == 200
    body = r.json()
    assert body["panels"] == []           # no per-panel boxes at cut-line stage
    assert len(body["lines"]) == 2        # the new field
    assert body["lines"][0]["orientation"] == "h"
