"""Tests for Qwen3.x thinking-mode handling in QwenVLM.ask_image.

Thinking mode (``enable_thinking=True``) and ``response_format=json_object``
are mutually exclusive on DashScope — when both are requested we must honor
thinking and drop json_mode, relying on the caller's tolerant parser. These
tests stub the OpenAI client so no network call is made.
"""
from __future__ import annotations

import json
from types import SimpleNamespace

from app.vlm.qwen import QwenVLM


def _make_vlm(*, enable_thinking: bool = False) -> QwenVLM:
    """Build a QwenVLM with a stubbed OpenAI client (no key needed)."""
    vlm = object.__new__(QwenVLM)  # bypass __init__ (which needs a real key)
    vlm._client = None  # type: ignore[attr-defined]
    vlm._model = "qwen3.7-plus"
    vlm._enable_thinking = enable_thinking
    return vlm


def _stub_client(vlm: QwenVLM, captured: dict):
    """Replace the OpenAI client so .chat.completions.create records its kwargs."""

    class _Create:
        def __init__(self, cap):
            self.cap = cap

        def create(self, **kwargs):
            self.cap["kwargs"] = kwargs
            # Return a minimal shape matching openai's response.
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content=json.dumps({"category": "x"}))
                    )
                ]
            )

    class _Completions:
        def __init__(self, cap):
            self._create = _Create(cap)

        def create(self, **kwargs):
            return self._create.create(**kwargs)

    class _Chat:
        def __init__(self, cap):
            self.completions = _Completions(cap)

    vlm._client = SimpleNamespace(chat=_Chat(captured))  # type: ignore[attr-defined]


def test_json_mode_on_thinking_off_sets_response_format():
    vlm = _make_vlm(enable_thinking=False)
    cap: dict = {}
    _stub_client(vlm, cap)
    vlm.ask_image("data:image/jpeg;base64,AAAA", "p", json_mode=True)
    kw = cap["kwargs"]
    assert kw["response_format"] == {"type": "json_object"}
    assert "extra_body" not in kw


def test_thinking_on_drops_json_mode():
    vlm = _make_vlm(enable_thinking=True)
    cap: dict = {}
    _stub_client(vlm, cap)
    vlm.ask_image("data:image/jpeg;base64,AAAA", "p", json_mode=True)
    kw = cap["kwargs"]
    # thinking + json are mutually exclusive: thinking wins, json_mode dropped
    assert kw["extra_body"] == {"enable_thinking": True}
    assert "response_format" not in kw


def test_thinking_explicit_override_beats_provider_default():
    # provider default is thinking=False, but caller forces it on
    vlm = _make_vlm(enable_thinking=False)
    cap: dict = {}
    _stub_client(vlm, cap)
    vlm.ask_image("data:image/jpeg;base64,AAAA", "p", json_mode=True, enable_thinking=True)
    kw = cap["kwargs"]
    assert kw["extra_body"] == {"enable_thinking": True}
    assert "response_format" not in kw


def test_thinking_off_explicit_override():
    # provider default is thinking=True, but caller forces it off → json works
    vlm = _make_vlm(enable_thinking=True)
    cap: dict = {}
    _stub_client(vlm, cap)
    vlm.ask_image("data:image/jpeg;base64,AAAA", "p", json_mode=True, enable_thinking=False)
    kw = cap["kwargs"]
    assert kw["response_format"] == {"type": "json_object"}
    assert "extra_body" not in kw


def test_no_json_no_thinking_sends_neither():
    vlm = _make_vlm(enable_thinking=False)
    cap: dict = {}
    _stub_client(vlm, cap)
    vlm.ask_image("data:image/jpeg;base64,AAAA", "p")
    kw = cap["kwargs"]
    assert "response_format" not in kw
    assert "extra_body" not in kw


def test_default_model_is_qwen3_7_plus():
    # Regression: ensure the code default isn't the deprecated qwen-vl-max.
    # Read the field default directly (not an instantiated Settings, which
    # would pick up whatever happens to be in the project .env on this machine).
    from app.config import Settings

    assert Settings.model_fields["vlm_model"].default == "qwen3.7-plus"
    assert Settings.model_fields["vlm_enable_thinking"].default is False
