"""Tests for the AI Native agent layer.

No network / no real models: the reasoning LLM and tools are monkeypatched or
stubbed. Mirrors the style of test_understanding.py — pure logic tests.
"""
from __future__ import annotations

import io
import json

import numpy as np
from PIL import Image

from app.agent import core as core_mod
from app.agent import tools as tools_mod
from app.agent.core import run_agent
from app.agent.llm import _parse_arguments
from app.agent.schemas import ToolCall
from app.agent.tools import ToolRegistry
from app.config import Settings


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


class _FakePipeline:
    """Minimal pipeline stub for tool tests — only the lazy loaders are used."""
    def __init__(self, vlm=None, ocr=None):
        self._vlm = vlm
        self._ocr = ocr

    def _get_vlm(self):
        return self._vlm

    def _get_ocr(self):
        return self._ocr


class _FakeVLM:
    """Stub VLM: returns canned text for ask_image."""
    def __init__(self, reply: str):
        self._reply = reply
        self.last_model = None

    def ask_image(self, data_url, prompt, *, max_tokens=1024, json_mode=False,
                  enable_thinking=None, model_override=None):
        self.last_model = model_override  # verify agent pins its VLM model
        return self._reply, 0.8


class _FakeOCR:
    """Stub OCR: returns one canned detection in tile-local coords."""
    def __init__(self, detections):
        self._dets = detections

    def detect_and_recognize(self, tile, **kwargs):
        return list(self._dets)


class _Det:
    """Mimics TextDetection (duck-typed)."""
    def __init__(self, polygon, text, confidence=0.9):
        self.polygon = polygon
        self.text = text
        self.confidence = confidence
        self.granularity = "line"
        self.lines = None


# ---------------------------------------------------------------------------
# ToolRegistry: geometry + tools
# ---------------------------------------------------------------------------


def _registry(arr=None, vlm=None, ocr=None):
    arr = arr if arr is not None else _solid(400, 300)
    h, w = arr.shape[:2]
    s = Settings()
    return ToolRegistry(arr, w, h, _FakePipeline(vlm=vlm, ocr=ocr), s)


def test_crop_norm_whole_image():
    reg = _registry(_solid(400, 300))
    crop, box = reg._crop_norm(None)
    assert box == [0, 0, 400, 300]
    assert crop.shape == (300, 400, 3)


def test_crop_norm_region_clamps_and_offsets():
    reg = _registry(_solid(400, 300))
    # region [0.1, 0.2, 0.5, 0.4] of 400x300 → x0=40,y0=60,x1=240,y1=180
    crop, box = reg._crop_norm([0.1, 0.2, 0.5, 0.4])
    assert box == [40, 60, 240, 180]
    assert crop.shape == (120, 200, 3)


def test_crop_norm_clamps_negative_and_overflow():
    reg = _registry(_solid(400, 300))
    crop, box = reg._crop_norm([-0.1, -0.1, 1.5, 1.5])  # overflow both axes
    assert box == [0, 0, 400, 300]  # clamped to image bounds


def test_crop_norm_rejects_empty_region():
    reg = _registry(_solid(400, 300))
    import pytest
    with pytest.raises(ValueError):
        reg._crop_norm([0.5, 0.5, 0.0, 0.0])  # zero-size


def test_tool_look_returns_description_and_pins_vlm_model():
    vlm = _FakeVLM("一张饮料盒展开图")
    reg = _registry(_solid(400, 300), vlm=vlm)
    result = reg.look(region=[0.1, 0.1, 0.5, 0.5], focus="品类")
    assert result["description"] == "一张饮料盒展开图"
    assert result["region_pixel"] == [40, 30, 240, 180]
    # The agent pins its own VLM model via settings.agent_vlm_model.
    assert vlm.last_model == Settings().agent_vlm_model


def test_tool_look_whole_image():
    vlm = _FakeVLM("整图概览")
    reg = _registry(_solid(400, 300), vlm=vlm)
    result = reg.look()  # no region → whole image
    assert result["region_norm"] is None
    assert result["region_pixel"] == [0, 0, 400, 300]


def test_tool_ocr_text_offsets_to_global_coords():
    # OCR returns tile-local coords; the tool must offset them to global.
    local_det = _Det(polygon=[[10, 20], [60, 20], [60, 40], [10, 40]], text="净含量")
    ocr = _FakeOCR([local_det])
    reg = _registry(_solid(400, 300), ocr=ocr)
    # region origin at (100, 60) → local (10,20) becomes global (110, 80)
    result = reg.ocr_text(region=[0.25, 0.2, 0.25, 0.4])
    assert result["count"] == 1
    t = result["texts"][0]
    assert t["text"] == "净含量"
    # global bbox = local bbox (10,20)-(60,40) offset by (100,60)
    assert t["bbox"] == [110, 80, 160, 100]


def test_tool_describe_returns_answer():
    vlm = _FakeVLM("左上角有品牌 logo")
    reg = _registry(_solid(400, 300), vlm=vlm)
    result = reg.describe(region=[0.05, 0.05, 0.2, 0.15], question="有 logo 吗?")
    assert result["answer"] == "左上角有品牌 logo"


def test_tool_dispatch_unknown_returns_error():
    reg = _registry()
    result = reg.dispatch("nonexistent_tool", {})
    assert "error" in result


def test_tool_dispatch_swallows_exceptions():
    class _BoomVLM:
        def ask_image(self, *a, **k):
            raise RuntimeError("VLM 挂了")
    reg = _registry(_solid(400, 300), vlm=_BoomVLM())
    # Go through dispatch() — that's the exception boundary; direct look() is
    # allowed to raise (only dispatch is called by the loop).
    result = reg.dispatch("look", {})
    assert "error" in result
    assert "VLM 挂了" in result["error"]


def test_tool_schemas_have_required_fields():
    reg = _registry()
    schemas = reg.schemas
    assert len(schemas) == 3
    names = {s["function"]["name"] for s in schemas}
    assert names == {"look", "ocr_text", "describe"}
    for s in schemas:
        assert s["type"] == "function"
        assert "parameters" in s["function"]


# ---------------------------------------------------------------------------
# _parse_arguments (DashScope double-JSON-encoding resilience)
# ---------------------------------------------------------------------------


def test_parse_arguments_dict_passthrough():
    assert _parse_arguments({"a": 1}) == {"a": 1}


def test_parse_arguments_none_empty():
    assert _parse_arguments(None) == {}
    assert _parse_arguments("") == {}


def test_parse_arguments_single_json_string():
    assert _parse_arguments('{"x": 1}') == {"x": 1}


def test_parse_arguments_double_json_string():
    # DashScope quirk: arguments is a JSON-encoded JSON string.
    assert _parse_arguments('"{\\"x\\": 1}"') == {"x": 1}


def test_parse_arguments_garbage_returns_empty():
    assert _parse_arguments("not json at all") == {}


# ---------------------------------------------------------------------------
# Agent loop (core.run_agent) — with a stubbed reasoning brain
# ---------------------------------------------------------------------------


class _ScriptedBrain:
    """Returns a canned sequence of (tool_calls, text) per step call."""
    def __init__(self, script):
        self.script = list(script)
        self.calls = 0

    def step(self, messages, tools_schema):
        idx = min(self.calls, len(self.script) - 1)
        self.calls += 1
        return self.script[idx]


def _patch_brain(monkeypatch, brain):
    # core.py does `from .llm import ReasoningLLM` — a name bound in core's
    # namespace — so patch it there, not on llm_mod.
    monkeypatch.setattr(core_mod, "ReasoningLLM", lambda settings: brain)


def test_agent_loop_runs_two_rounds_then_concludes(monkeypatch):
    # Round 1: look at whole image. Round 2: conclude with JSON.
    round1_tools = [ToolCall(id="c1", name="look", arguments={})]
    conclusion_json = json.dumps({
        "category": "食品-饮料", "category_confidence": 0.85,
        "panel_count_estimate": 6, "summary": "饮料盒展开图",
    })
    brain = _ScriptedBrain([
        (round1_tools, "先看整图"),
        (None, conclusion_json),
    ])
    _patch_brain(monkeypatch, brain)

    # Stub the tools so dispatch returns canned data.
    reg_instance = _registry(_solid(400, 300), vlm=_FakeVLM("概览"))
    monkeypatch.setattr(core_mod, "ToolRegistry", lambda *a, **k: reg_instance)

    result = run_agent(_png_bytes(_solid(400, 300)), Settings(), _FakePipeline())
    assert result.rounds == 2
    assert len(result.trace) == 2
    assert result.trace[0].tool_calls[0].name == "look"
    assert result.trace[1].tool_calls == []  # conclusion round
    assert result.conclusion.category == "食品-饮料"
    assert result.conclusion.category_confidence == 0.85
    assert result.fallback is False


def test_agent_loop_terminates_when_no_tools(monkeypatch):
    brain = _ScriptedBrain([(None, '{"category":"其他","summary":"x"}')])
    _patch_brain(monkeypatch, brain)
    reg_instance = _registry(_solid(200, 200))
    monkeypatch.setattr(core_mod, "ToolRegistry", lambda *a, **k: reg_instance)

    result = run_agent(_png_bytes(_solid(200, 200)), Settings(), _FakePipeline())
    assert result.rounds == 1
    assert result.conclusion.category == "其他"


def test_agent_records_observations_per_tool(monkeypatch):
    round1_tools = [
        ToolCall(id="c1", name="look", arguments={"focus": "品类"}),
        ToolCall(id="c2", name="describe", arguments={"region": [0, 0, 0.5, 0.5], "question": "啥?"}),
    ]
    brain = _ScriptedBrain([
        (round1_tools, "先看品类和左上区域"),
        (None, '{"category":"x","summary":"y"}'),
    ])
    _patch_brain(monkeypatch, brain)
    reg_instance = _registry(_solid(400, 300), vlm=_FakeVLM("ok"))
    monkeypatch.setattr(core_mod, "ToolRegistry", lambda *a, **k: reg_instance)

    result = run_agent(_png_bytes(_solid(400, 300)), Settings(), _FakePipeline())
    assert len(result.trace[0].observations) == 2
    assert "description" in result.trace[0].observations[0]
    assert "answer" in result.trace[0].observations[1]


def test_agent_falls_back_on_brain_failure(monkeypatch):
    class _BoomBrain:
        def step(self, *a, **k):
            raise RuntimeError("qwen3-max 不可用")
    monkeypatch.setattr(core_mod, "ReasoningLLM", lambda settings: _BoomBrain())
    reg_instance = _registry(_solid(200, 200))
    monkeypatch.setattr(core_mod, "ToolRegistry", lambda *a, **k: reg_instance)

    result = run_agent(_png_bytes(_solid(200, 200)), Settings(), _FakePipeline())
    assert result.fallback is True
    assert result.rounds == 0
    assert "qwen3-max 不可用" in (result.error or "")


def test_agent_falls_back_on_bad_image():
    result = run_agent(b"not an image", Settings(), _FakePipeline())
    assert result.fallback is True
    assert result.conclusion.category_confidence == 0.0
