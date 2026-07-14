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


def test_pipeline_vlm_fallback_skipped_by_default():
    """OCR path is PaddleOCR-only unless both VLM switches are enabled.

    Explicitly constructs a Settings with BOTH VLM switches off — don't rely on
    Settings() defaults, which read .env and can flip in a dev env where VLM is
    turned on (making this assertion wrong).
    """
    s = Settings().model_copy(update={
        "vlm_enabled": False, "vlm_ocr_fallback_enabled": False,
    })
    pipe = Pipeline(s)
    items = [
        Item(
            id="t1", type="text", text="?", polygon=[[10, 10], [30, 10], [30, 30], [10, 30]],
            bbox=[10, 10, 30, 30], confidence=0.1, source="paddleocr",
        ),
    ]
    out, n_crops = pipe._maybe_vlm_fallback(_img(), items)
    assert n_crops == 0
    assert out[0].source == "paddleocr"


def test_pipeline_vlm_fallback_uses_batch(monkeypatch):
    """_maybe_vlm_fallback collects suspects and calls the batched-with-prompts
    method once, passing each crop's prompt through."""
    pipe = Pipeline(Settings())

    # Force-enable both VLM switches and a high threshold so items qualify.
    s = pipe.settings.model_copy(update={
        "vlm_enabled": True,
        "vlm_ocr_fallback_enabled": True,
        "rec_confidence_fallback": 0.99,
    })
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

    calls = {"batch": 0, "crops": []}

    class _FakeVLM:
        def recognize_crops_with_prompts_batch(self, image, crops):
            calls["batch"] += 1
            calls["crops"] = crops
            assert len(crops) == 2  # only the two suspects
            return [("HELLO", 0.8), ("WORLD", 0.8)]

    monkeypatch.setattr(pipe, "_get_vlm", lambda: _FakeVLM())

    out, n_crops = pipe._maybe_vlm_fallback(_img(), items)

    assert calls["batch"] == 1          # ONE batched call, not 2 serial
    assert n_crops == 2
    # Each crop carries a (polygon, prompt) pair; prompts are non-empty strings.
    for poly, prompt in calls["crops"]:
        assert isinstance(prompt, str) and prompt
    # Suspects got VLM text + source tag; type stays text (not art_text).
    assert out[0].text == "HELLO" and out[0].type == "text"
    assert out[0].source == "vlm_fallback"
    assert out[1].text == "ok"   and out[1].type == "text"      # unchanged
    assert out[2].text == "WORLD" and out[2].type == "text"
    assert out[2].source == "vlm_fallback"


def test_pipeline_vlm_fallback_logs_sent_rescued_empty(monkeypatch, caplog):
    """The summary log line records how many crops were sent, how many the VLM
    rescued (non-empty text), and how many came back empty. This is the single
    line operators use to tell whether the VLM actually worked."""
    import logging

    pipe = Pipeline(Settings())
    s = pipe.settings.model_copy(update={
        "vlm_enabled": True,
        "vlm_ocr_fallback_enabled": True,
        "rec_confidence_fallback": 0.99,
        "circular_detect_enabled": False,  # keep this test about suspects only
    })
    pipe.settings = s

    # 3 suspects: 2 will be rescued, 1 comes back empty.
    items = [
        Item(id="t1", type="text", text="?", polygon=[[0, 0], [9, 0], [9, 9], [0, 9]],
             bbox=[0, 0, 9, 9], confidence=0.1, source="paddleocr"),
        Item(id="t2", type="text", text="?", polygon=[[0, 0], [9, 0], [9, 9], [0, 9]],
             bbox=[0, 0, 9, 9], confidence=0.2, source="paddleocr"),
        Item(id="t3", type="text", text="?", polygon=[[0, 0], [9, 0], [9, 9], [0, 9]],
             bbox=[0, 0, 9, 9], confidence=0.3, source="paddleocr"),
    ]

    class _FakeVLM:
        def recognize_crops_with_prompts_batch(self, image, crops):
            # 2 non-empty (rescued) + 1 empty (VLM couldn't read it).
            return [("GOOD1", 0.8), ("GOOD2", 0.8), ("", 0.0)]

    monkeypatch.setattr(pipe, "_get_vlm", lambda: _FakeVLM())

    with caplog.at_level(logging.INFO, logger="app.pipeline"):
        out, n_crops = pipe._maybe_vlm_fallback(_img(), items)

    assert n_crops == 3
    # Find the summary line and check its counts.
    summary = [r.message for r in caplog.records if "vlm fallback" in r.message]
    assert summary, f"no vlm fallback summary logged; got {[r.message for r in caplog.records]}"
    line = summary[-1]
    assert "sent=3" in line
    assert "rescued=2" in line
    assert "empty=1" in line
    assert "threshold=0.99" in line


# ---------------------------------------------------------------------------
# Pipeline._drop_low_confidence — /analyze confidence policy (drop < threshold)
# ---------------------------------------------------------------------------


def _text_item(id_, conf, source="paddleocr"):
    return Item(
        id=id_, type="text", text=f"t{id_}",
        polygon=[[0, 0], [10, 0], [10, 10], [0, 10]],
        bbox=[0, 0, 10, 10], confidence=conf, source=source,
    )


def _code_item(id_, conf, type_="qr"):
    return Item(
        id=id_, type=type_, content="payload",
        polygon=[[0, 0], [10, 0], [10, 10], [0, 10]],
        bbox=[0, 0, 10, 10], confidence=conf, source="pyzbar",
    )


def test_drop_low_confidence_drops_text_below_threshold():
    """Text items below rec_confidence_drop are removed."""
    s = Settings().model_copy(update={
        "rec_confidence_drop": 0.60,
        "rec_confidence_fallback": 0.94,
    })
    pipe = Pipeline(s)
    items = [
        _text_item("t1", 0.55),   # below 0.60 → dropped
        _text_item("t2", 0.60),   # exactly 0.60 → kept (>=)
        _text_item("t3", 0.80),   # kept
    ]
    kept = pipe._drop_low_confidence(items)
    kept_ids = {it.id for it in kept}
    assert kept_ids == {"t2", "t3"}


def test_drop_low_confidence_keeps_codes_regardless_of_confidence():
    """qr/barcode are NEVER dropped even at low confidence (different semantics)."""
    s = Settings().model_copy(update={
        "rec_confidence_drop": 0.60,
        "rec_confidence_fallback": 0.94,
    })
    pipe = Pipeline(s)
    items = [
        _text_item("t1", 0.50),         # dropped
        _code_item("q1", 0.20, "qr"),   # kept
        _code_item("b1", 0.10, "barcode"),  # kept
    ]
    kept = pipe._drop_low_confidence(items)
    kept_ids = {it.id for it in kept}
    assert kept_ids == {"q1", "b1"}


def test_drop_low_confidence_rescued_by_vlm_survives():
    """An item the VLM lifted above the drop threshold (source=vlm_fallback)
    must survive — the policy runs AFTER the VLM pass."""
    s = Settings().model_copy(update={
        "rec_confidence_drop": 0.60,
        "rec_confidence_fallback": 0.94,
    })
    pipe = Pipeline(s)
    items = [
        _text_item("t1", 0.70, source="vlm_fallback"),  # rescued above 0.60
        _text_item("t2", 0.55, source="vlm_fallback"),  # VLM couldn't lift it
    ]
    kept = pipe._drop_low_confidence(items)
    assert {it.id for it in kept} == {"t1"}


def test_drop_low_confidence_clamped_when_drop_above_fallback():
    """Misconfiguration (drop > fallback) is clamped to the fallback value so
    the re-read set isn't silently widened."""
    s = Settings().model_copy(update={
        "rec_confidence_drop": 0.99,       # misconfigured
        "rec_confidence_fallback": 0.60,
    })
    pipe = Pipeline(s)
    items = [
        _text_item("t1", 0.50),
        _text_item("t2", 0.80),  # would be dropped if drop=0.99 were honored
    ]
    kept = pipe._drop_low_confidence(items)
    assert {it.id for it in kept} == {"t2"}

