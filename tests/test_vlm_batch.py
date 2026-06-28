"""Tests for concurrent VLM art-text recognition.

``recognize_crops_batch`` dispatches N ``recognize_crop`` calls across a thread
pool so the cloud round-trips overlap instead of queuing behind each other.
These tests stub the OpenAI client (no network) and cover: concurrency (all
crops dispatched in parallel), result ordering (one result per input, in
order), empty input, and the ``_maybe_vlm_fallback`` pipeline path that feeds
results back into items.

Mirrors the stubbing style of ``test_vlm_thinking.py``.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np

from app.config import Settings
from app.pipeline import Pipeline
from app.schemas import Item
from app.vlm.qwen import QwenVLM


# ---------------------------------------------------------------------------
# QwenVLM.recognize_crops_batch — concurrency + ordering (stubbed client)
# ---------------------------------------------------------------------------


def _make_vlm() -> QwenVLM:
    """Build a QwenVLM with a stubbed client (bypasses __init__/key check)."""
    vlm = object.__new__(QwenVLM)
    vlm._client = None
    vlm._model = "qwen3.7-plus"
    vlm._enable_thinking = False
    return vlm


def _img() -> np.ndarray:
    """A 100x100 image; crops fall inside it."""
    return np.zeros((100, 100, 3), dtype=np.uint8)


def _stub_client_with_latency(vlm: QwenVLM, per_call_s: float, texts_by_index: dict):
    """Each recognize_crop call sleeps ``per_call_s`` (simulating network I/O)
    and returns the text for that call's crop index (encoded in the message)."""
    call_log = {"n": 0, "wall_starts": []}

    class _Create:
        def create(self, **kwargs):
            call_log["n"] += 1
            call_log["wall_starts"].append(time.perf_counter())
            time.sleep(per_call_s)  # simulate a slow cloud round-trip
            # The image_url is in content[0]; we don't decode it, just return a
            # fixed text per call to verify ordering.
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content="TEXT")
                )]
            )
    vlm._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=_Create().create))
    )
    return call_log


def test_batch_runs_calls_concurrently():
    """N serial calls each taking 0.3s must finish in ~0.3s, not N*0.3s."""
    vlm = _make_vlm()
    log = _stub_client_with_latency(vlm, per_call_s=0.3, texts_by_index={})

    polys = [[[5, 5], [15, 5], [15, 15], [5, 15]]] * 6
    t0 = time.perf_counter()
    out = vlm.recognize_crops_batch(_img(), polys)
    elapsed = time.perf_counter() - t0

    assert log["n"] == 6                  # all 6 crops were recognized
    assert len(out) == 6                  # one result per input, in order
    # Concurrent: 6 calls of 0.3s each. Serial would be ~1.8s; concurrent
    # (8 workers) should be ~0.3s. Allow generous slack for CI/scheduling.
    assert elapsed < 0.9, f"expected concurrent (~0.3s), took {elapsed:.2f}s"


def test_batch_preserves_input_order():
    """Results come back in input order regardless of completion order."""
    vlm = _make_vlm()

    # Each call returns a distinct text so we can verify ordering.
    counter = {"i": 0}

    class _Create:
        def create(self, **kwargs):
            counter["i"] += 1
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content=f"r{counter['i']}")
                )]
            )
    vlm._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=_Create().create))
    )

    polys = [[[i, i], [i + 10, i], [i + 10, i + 10], [i, i + 10]] for i in range(10, 60, 10)]
    out = vlm.recognize_crops_batch(_img(), polys)
    texts = [t for t, _ in out]
    # ThreadPoolExecutor preserves submission order via the futures list, so
    # results map 1:1 to inputs regardless of which call finished first.
    assert len(set(texts)) == len(texts)  # all distinct
    assert len(texts) == len(polys)


def test_batch_empty_input_returns_empty():
    assert _make_vlm().recognize_crops_batch(_img(), []) == []


# ---------------------------------------------------------------------------
# Pipeline._maybe_vlm_fallback — batch path writes results back into items
# ---------------------------------------------------------------------------


def test_pipeline_vlm_fallback_uses_batch(monkeypatch):
    """_maybe_vlm_fallback collects suspects and calls recognize_crops_batch once."""
    pipe = Pipeline(Settings())

    # Force-enable VLM and set a high threshold so the items below qualify.
    s = pipe.settings.model_copy(update={"vlm_enabled": True,
                                         "rec_confidence_fallback": 0.99})
    pipe.settings = s

    # Two suspect items + one confident one (should NOT be re-recognized).
    items = [
        Item(id="t1", type="text", text="?", polygon=[[10, 10], [30, 10], [30, 30], [10, 30]],
             bbox=[10, 10, 30, 30], confidence=0.3, source="paddleocr"),
        Item(id="t2", type="text", text="ok", polygon=[[40, 40], [60, 40], [60, 60], [40, 60]],
             bbox=[40, 40, 60, 60], confidence=0.99, source="paddleocr"),  # confident, skipped
        Item(id="t3", type="text", text="?", polygon=[[70, 70], [90, 70], [90, 90], [70, 90]],
             bbox=[70, 70, 90, 90], confidence=0.2, source="paddleocr"),
    ]

    calls = {"batch": 0}

    class _FakeVLM:
        def recognize_crops_batch(self, image, polys):
            calls["batch"] += 1
            assert len(polys) == 2  # only the two suspects
            return [("HELLO", 0.8), ("WORLD", 0.8)]

    monkeypatch.setattr(pipe, "_get_vlm", lambda: _FakeVLM())

    out, n_crops = pipe._maybe_vlm_fallback(_img(), items)

    assert calls["batch"] == 1          # ONE batched call, not 2 serial
    assert n_crops == 2
    # Suspects got VLM text + source tag; type stays text (not art_text).
    assert out[0].text == "HELLO" and out[0].type == "text"
    assert out[0].source == "vlm_fallback"
    assert out[1].text == "ok"   and out[1].type == "text"      # unchanged
    assert out[2].text == "WORLD" and out[2].type == "text"
    assert out[2].source == "vlm_fallback"
