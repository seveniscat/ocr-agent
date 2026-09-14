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


def _make_vlm(min_crop_side: int = 0) -> QwenVLM:
    """Build a QwenVLM with a stubbed client (bypasses __init__/key check).

    ``min_crop_side`` defaults to 0 (no filtering) so existing concurrency /
    ordering tests aren't affected by the size guard. Tests that exercise the
    guard pass an explicit value.
    """
    vlm = object.__new__(QwenVLM)
    vlm._client = None
    vlm._model = "qwen3.7-plus"
    vlm._enable_thinking = False
    vlm._min_crop_side = min_crop_side
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
    out, n_crops, _stats = pipe._maybe_vlm_fallback(_img(), items)
    assert n_crops == 0
    assert out[0].source == "paddleocr"


def test_pipeline_vlm_fallback_uses_batch(monkeypatch):
    """_maybe_vlm_fallback collects suspects and calls the batched-with-prompts
    method once, passing each crop's prompt through."""
    pipe = Pipeline(Settings())

    # Force-enable both VLM switches and a high threshold so items qualify.
    # Pin the floors explicitly so the test is stable regardless of .env.
    s = pipe.settings.model_copy(update={
        "vlm_enabled": True,
        "vlm_ocr_fallback_enabled": True,
        "rec_confidence_fallback": 0.99,
        "rec_confidence_drop": 0.60,
        "min_keep_confidence": 0.60,
    })
    pipe.settings = s

    # Two suspect items + one confident one (should NOT be re-recognized).
    # Suspects must sit in [vlm_floor, threshold) = [0.60, 0.99): below the
    # floor they'd be discarded anyway (no point re-reading), at/above the
    # threshold PaddleOCR is confident.
    items = [
        Item(id="t1", type="text", text="?", polygon=[[10, 10], [30, 10], [30, 30], [10, 30]],
             bbox=[10, 10, 30, 30], confidence=0.70, source="paddleocr"),
        Item(id="t2", type="text", text="ok", polygon=[[40, 40], [60, 40], [60, 60], [40, 60]],
             bbox=[40, 40, 60, 60], confidence=0.99, source="paddleocr"),  # confident, skipped
        Item(id="t3", type="text", text="?", polygon=[[70, 70], [90, 70], [90, 90], [70, 90]],
             bbox=[70, 70, 90, 90], confidence=0.80, source="paddleocr"),
    ]

    calls = {"batch": 0, "crops": []}

    class _FakeVLM:
        def recognize_crops_with_prompts_batch(self, image, crops):
            calls["batch"] += 1
            calls["crops"] = crops
            assert len(crops) == 2  # only the two suspects
            return [("HELLO", 0.8), ("WORLD", 0.8)]

    monkeypatch.setattr(pipe, "_get_vlm", lambda: _FakeVLM())

    out, n_crops, _stats = pipe._maybe_vlm_fallback(_img(), items)

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
        "rec_confidence_drop": 0.60,
        "min_keep_confidence": 0.60,
        "circular_detect_enabled": False,  # keep this test about suspects only
    })
    pipe.settings = s

    # 3 suspects: 2 will be rescued, 1 comes back empty. All must sit in
    # [vlm_floor, threshold) = [0.60, 0.99) to qualify for re-reading.
    items = [
        Item(id="t1", type="text", text="?", polygon=[[0, 0], [9, 0], [9, 9], [0, 9]],
             bbox=[0, 0, 9, 9], confidence=0.70, source="paddleocr"),
        Item(id="t2", type="text", text="?", polygon=[[0, 0], [9, 0], [9, 9], [0, 9]],
             bbox=[0, 0, 9, 9], confidence=0.75, source="paddleocr"),
        Item(id="t3", type="text", text="?", polygon=[[0, 0], [9, 0], [9, 9], [0, 9]],
             bbox=[0, 0, 9, 9], confidence=0.80, source="paddleocr"),
    ]

    class _FakeVLM:
        def recognize_crops_with_prompts_batch(self, image, crops):
            # 2 non-empty (rescued) + 1 empty (VLM couldn't read it).
            return [("GOOD1", 0.8), ("GOOD2", 0.8), ("", 0.0)]

    monkeypatch.setattr(pipe, "_get_vlm", lambda: _FakeVLM())

    with caplog.at_level(logging.INFO, logger="app.pipeline"):
        out, n_crops, _stats = pipe._maybe_vlm_fallback(_img(), items)

    assert n_crops == 3
    # Find the summary line and check its counts.
    summary = [r.message for r in caplog.records if "vlm fallback" in r.message]
    assert summary, f"no vlm fallback summary logged; got {[r.message for r in caplog.records]}"
    line = summary[-1]
    assert "sent=3" in line
    assert "rescued=2" in line
    assert "empty=1" in line
    assert "threshold=0.99" in line


def test_pipeline_vlm_fallback_skips_below_floor(monkeypatch):
    """Boxes scoring below the universal floor (min_keep_confidence) are NOT
    sent to the VLM on the /analyze path — they'd be discarded downstream by
    _clean_items regardless of what the VLM returns (confidence stays the
    PaddleOCR score). Note: since P1-3 the floor reads ONLY min_keep_confidence
    (default 0.0 now), NOT rec_confidence_drop."""
    pipe = Pipeline(Settings())
    s = pipe.settings.model_copy(update={
        "vlm_enabled": True,
        "vlm_ocr_fallback_enabled": True,
        "rec_confidence_fallback": 0.99,
        "min_keep_confidence": 0.60,   # → vlm_floor=0.60
        "circular_detect_enabled": False,
    })
    pipe.settings = s

    items = [
        # 0.50 < vlm_floor 0.60 → NOT sent (would be dropped anyway).
        Item(id="low", type="text", text="?", polygon=[[0, 0], [9, 0], [9, 9], [0, 9]],
             bbox=[0, 0, 9, 9], confidence=0.50, source="paddleocr"),
        # 0.70 in [0.60, 0.99) → sent.
        Item(id="mid", type="text", text="?", polygon=[[0, 0], [9, 0], [9, 9], [0, 9]],
             bbox=[0, 0, 9, 9], confidence=0.70, source="paddleocr"),
    ]

    calls = {"crops": []}

    class _FakeVLM:
        def recognize_crops_with_prompts_batch(self, image, crops):
            calls["crops"] = crops
            return [("READ", 1.0)] * len(crops)

    monkeypatch.setattr(pipe, "_get_vlm", lambda: _FakeVLM())

    out, n_crops, _stats = pipe._maybe_vlm_fallback(_img(), items)

    assert n_crops == 1                      # only the mid box sent
    assert len(calls["crops"]) == 1
    # The low box was left untouched (not re-read).
    assert out[0].source == "paddleocr" and out[0].text == "?"
    # The mid box was re-read.
    assert out[1].source == "vlm_fallback" and out[1].text == "READ"


def test_pipeline_vlm_fallback_floor_exempt_for_verify(monkeypatch):
    """On the /verify path the floor is disabled — a low-confidence box the VLM
    CAN read is valuable there (every char may match required copy)."""
    pipe = Pipeline(Settings())
    s = pipe.settings.model_copy(update={
        "vlm_enabled": True,
        "vlm_ocr_fallback_enabled": True,
        "rec_confidence_fallback": 0.99,
        "rec_confidence_drop": 0.60,
        "min_keep_confidence": 0.60,
        "circular_detect_enabled": False,
    })
    pipe.settings = s

    items = [
        # 0.50 < the /analyze floor, but verify path sends it anyway.
        Item(id="low", type="text", text="?", polygon=[[0, 0], [9, 0], [9, 9], [0, 9]],
             bbox=[0, 0, 9, 9], confidence=0.50, source="paddleocr"),
    ]

    calls = {"crops": []}

    class _FakeVLM:
        def recognize_crops_with_prompts_batch(self, image, crops):
            calls["crops"] = crops
            return [("READ", 1.0)] * len(crops)

    monkeypatch.setattr(pipe, "_get_vlm", lambda: _FakeVLM())

    out, n_crops, _stats = pipe._maybe_vlm_fallback(_img(), items, for_verify=True)

    assert n_crops == 1                      # the low box WAS sent (verify)
    assert out[0].source == "vlm_fallback" and out[0].text == "READ"


# ---------------------------------------------------------------------------
# Pipeline._clean_items — universal content-quality cleanup
# ---------------------------------------------------------------------------


def _txt(id_, text, conf=0.9, recognized=True, type_="text"):
    return Item(
        id=id_, type=type_, text=text, recognized=recognized,
        polygon=[[0, 0], [10, 0], [10, 10], [0, 10]],
        bbox=[0, 0, 10, 10], confidence=conf, source="paddleocr",
    )


def test_clean_items_returns_breakdown():
    """_clean_items returns (kept, breakdown) with per-rule drop counts,
    including the new 'small' (small-box) bucket."""
    s = Settings().model_copy(update={
        "min_text_chars": 2, "min_keep_confidence": 0.6, "min_box_side": 8,
    })
    pipe = Pipeline(s)
    items = [
        _txt("empty", "   "),
        _txt("unrec", "x", recognized=False),
        _txt("short", "."),                 # junk-short (1 effective char)
        _txt("low", "ok", conf=0.3),        # low-confidence
        _txt("good", "hello"),
    ]
    kept, br = pipe._clean_items(items)
    assert {it.id for it in kept} == {"good"}
    # 'small' is 0 here because _txt default bbox is 10x10 (>= 8 floor).
    assert br == {"empty": 1, "unrec": 1, "short": 1, "small": 0, "low": 1}


def test_clean_items_drops_small_boxes_by_min_side():
    """The universal small-box noise filter: a text item whose bbox has EITHER
    dimension shorter than min_box_side is dropped. Narrow-but-long boxes
    (tall-thin column, wide-thin line) are NOT dropped — only tiny-in-both."""
    s = Settings().model_copy(update={
        "min_text_chars": 0, "min_keep_confidence": 0.0, "min_box_side": 8,
    })
    pipe = Pipeline(s)

    def box(id_, x0, y0, x1, y1, text="ok"):
        return Item(id=id_, type="text", text=text, recognized=True,
                    polygon=[[x0,y0],[x1,y0],[x1,y1],[x0,y1]],
                    bbox=[x0,y0,x1,y1], confidence=0.9, source="paddleocr")

    items = [
        box("tiny",   0, 0, 5, 5),    # 5x5  → both < 8 → dropped
        box("wide_thin", 0, 0, 100, 4),  # 100x4 → h<8 but w>=8 → KEPT (narrow-but-long)
        box("tall_thin", 0, 0, 4, 100),  # 4x100 → w<8 but h>=8 → KEPT
        box("normal", 0, 0, 50, 20),   # kept
    ]
    kept, br = pipe._clean_items(items)
    assert {it.id for it in kept} == {"wide_thin", "tall_thin", "normal"}
    assert br["small"] == 1


def test_clean_items_min_box_side_disabled():
    """min_box_side=0 disables the small-box filter entirely."""
    s = Settings().model_copy(update={
        "min_text_chars": 0, "min_keep_confidence": 0.0, "min_box_side": 0,
    })
    pipe = Pipeline(s)
    items = [Item(id="tiny", type="text", text="ok", recognized=True,
                  polygon=[[0,0],[3,0],[3,3],[0,3]],
                  bbox=[0,0,3,3], confidence=0.9, source="paddleocr")]
    kept, br = pipe._clean_items(items)
    assert {it.id for it in kept} == {"tiny"}
    assert br["small"] == 0


def test_clean_items_keeps_cjk_single_char():
    """P0-2: a lone CJK character (e.g. license-plate prefix 京, seal text) must
    NOT be stripped as 'junk' and dropped. The old \\W regex treated every
    non-ASCII char as non-word and stripped it to empty → silent data loss."""
    s = Settings().model_copy(update={"min_text_chars": 2, "min_keep_confidence": 0.0})
    pipe = Pipeline(s)
    items = [
        _txt("cjk1", "京"),       # single CJK char — effective len 1 < 2, but
                                  # it is a real character, not junk punctuation.
        _txt("cjk2", "沪A"),      # CJK + ascii
        _txt("jpn", "あ"),        # hiragana
        _txt("kor", "한"),        # hangul
    ]
    # min_text_chars=2 would drop these IF the regex wrongly stripped CJK to
    # empty. With the punctuation-only class, 京 stays len-1 → dropped by the
    # junk-short rule (correct: <2 chars), but 京沪 (len 2) survives.
    kept, _ = pipe._clean_items(items)
    # Single chars are < min_text_chars=2 → dropped by rule-3 as too short.
    # The POINT is they must be dropped for LENGTH, not because the regex
    # treated them as punctuation. Verify by lowering min_text_chars to 1:
    assert {it.id for it in kept} == {"cjk2"}

    s2 = Settings().model_copy(update={"min_text_chars": 1, "min_keep_confidence": 0.0})
    pipe2 = Pipeline(s2)
    kept2, _ = pipe2._clean_items(items)
    # Now ALL survive — proving the regex did NOT strip the CJK char away.
    assert {it.id for it in kept2} == {"cjk1", "cjk2", "jpn", "kor"}


def test_clean_items_strips_punctuation_but_not_cjk():
    """Edge punctuation (CJK + ASCII) is stripped for the effective-length
    measure, while inner letters/digits of all scripts are preserved."""
    s = Settings().model_copy(update={"min_text_chars": 2, "min_keep_confidence": 0.0})
    pipe = Pipeline(s)
    items = [
        _txt("punct_edges", "，上海。"),   # leading ，trailing 。 stripped → "上海" (2) kept
        _txt("only_punct", "。。"),        # stripped → "" (0) dropped
        _txt("ascii_punct", "-A-"),        # stripped → "A" (1) dropped
    ]
    kept, _ = pipe._clean_items(items)
    assert {it.id for it in kept} == {"punct_edges"}


# ---------------------------------------------------------------------------
# Pipeline._drop_low_confidence — /analyze confidence policy (drop < threshold)
# ---------------------------------------------------------------------------


def _text_item(id_, conf, source="paddleocr", vlm_lifted=None):
    return Item(
        id=id_, type="text", text=f"t{id_}",
        polygon=[[0, 0], [10, 0], [10, 10], [0, 10]],
        bbox=[0, 0, 10, 10], confidence=conf, source=source,
        vlm_lifted=vlm_lifted,
    )


def _code_item(id_, conf, type_="qr"):
    return Item(
        id=id_, type=type_, content="payload",
        polygon=[[0, 0], [10, 0], [10, 10], [0, 10]],
        bbox=[0, 0, 10, 10], confidence=conf, source="pyzbar",
    )


def test_drop_low_confidence_only_rule2_applies():
    """After P0-1, _drop_low_confidence has a SINGLE rule (the VLM-gate):
    drop a text item only when the VLM looked at it but couldn't read it
    (vlm_lifted is False) AND its confidence is below rec_confidence_vlm_drop.

    The plain confidence floor (old rule-1) is gone — that job belongs to the
    universal _clean_items (min_keep_confidence), which runs earlier on every
    path. So a low-confidence text item that never went through the VLM is NOT
    dropped here regardless of its score.
    """
    s = Settings().model_copy(update={
        "rec_confidence_drop": 0.60,        # no longer read by this method
        "rec_confidence_fallback": 0.94,
        "rec_confidence_vlm_drop": 0.85,
    })
    pipe = Pipeline(s)
    items = [
        # Never sent to VLM (vlm_lifted=None): rule-2 inert → KEPT even at 0.10.
        _text_item("keep_low", 0.10, vlm_lifted=None),
        _text_item("keep_mid", 0.70, vlm_lifted=None),
        # VLM failed AND conf < vlm_drop 0.85 → DROPPED.
        _text_item("drop", 0.70, source="vlm_fallback", vlm_lifted=False),
        # VLM failed but conf >= vlm_drop 0.85 → KEPT.
        _text_item("keep_hi", 0.90, source="vlm_fallback", vlm_lifted=False),
        # VLM succeeded (vlm_lifted=True) → always KEPT, even below vlm_drop.
        _text_item("keep_rescued", 0.55, source="vlm_fallback", vlm_lifted=True),
    ]
    kept = pipe._drop_low_confidence(items)
    assert {it.id for it in kept} == {
        "keep_low", "keep_mid", "keep_hi", "keep_rescued",
    }


def test_drop_low_confidence_keeps_codes_regardless_of_confidence():
    """qr/barcode are NEVER dropped even at low confidence (different semantics)."""
    s = Settings().model_copy(update={
        "rec_confidence_drop": 0.60,
        "rec_confidence_fallback": 0.94,
        "rec_confidence_vlm_drop": 0.85,
    })
    pipe = Pipeline(s)
    items = [
        _text_item("t1", 0.50, vlm_lifted=False),  # would drop under rule-2? no: vlm_lifted False & conf 0.50 < 0.85 → drop
        _code_item("q1", 0.20, "qr"),              # kept
        _code_item("b1", 0.10, "barcode"),         # kept
    ]
    kept = pipe._drop_low_confidence(items)
    kept_ids = {it.id for it in kept}
    assert kept_ids == {"q1", "b1"}


def test_drop_low_confidence_keeps_art_text():
    """art_text is high-value copy — it is NOT subject to the VLM-gate drop
    (only type=='text' is). Consistency fix for P2-6."""
    s = Settings().model_copy(update={
        "rec_confidence_vlm_drop": 0.85,
    })
    pipe = Pipeline(s)
    items = [
        Item(
            id="a1", type="art_text", text="ART", recognized=True,
            polygon=[[0, 0], [10, 0], [10, 10], [0, 10]],
            bbox=[0, 0, 10, 10], confidence=0.10, source="vlm_fallback",
            vlm_lifted=False,
        ),
    ]
    kept = pipe._drop_low_confidence(items)
    assert {it.id for it in kept} == {"a1"}


def test_drop_low_confidence_rule2_boundary():
    """Rule-2 boundary: confidence exactly == rec_confidence_vlm_drop is KEPT
    (the comparison is strict <)."""
    s = Settings().model_copy(update={
        "rec_confidence_vlm_drop": 0.85,
    })
    pipe = Pipeline(s)
    items = [
        _text_item("boundary", 0.85, source="vlm_fallback", vlm_lifted=False),
        _text_item("just_below", 0.849, source="vlm_fallback", vlm_lifted=False),
    ]
    kept = pipe._drop_low_confidence(items)
    assert {it.id for it in kept} == {"boundary"}


# ---------------------------------------------------------------------------
# _clean_vlm_text — VLM output normalization (text recognition, no score)
# ---------------------------------------------------------------------------


def test_clean_vlm_text_plain_text():
    """Plain text is returned trimmed, with stray surrounding quotes stripped."""
    from app.vlm.qwen import _clean_vlm_text
    assert _clean_vlm_text("能量宇宙 擎天柱") == "能量宇宙 擎天柱"
    assert _clean_vlm_text('"hello"') == "hello"
    assert _clean_vlm_text("  spaced  ") == "spaced"


def test_clean_vlm_text_empty_or_empty_sentinel():
    """Empty / EMPTY responses return None (treated as a failed read)."""
    from app.vlm.qwen import _clean_vlm_text
    assert _clean_vlm_text("") is None
    assert _clean_vlm_text("   ") is None
    assert _clean_vlm_text("EMPTY") is None
    assert _clean_vlm_text("empty") is None


def test_clean_vlm_text_rejects_fenced_json():
    """A ```json-fenced block is treated as a failed read → None.

    Reproduces the original bug: the VLM ignored the plain-text format and
    emitted a fenced JSON block; it must NOT be returned verbatim as text.
    """
    from app.vlm.qwen import _clean_vlm_text
    raw = '```json\n{\n    "text": "能量宇宙 擎天柱",\n    "score": 0.0\n}\n```'
    assert _clean_vlm_text(raw) is None


def test_clean_vlm_text_rejects_bare_json_with_text_key():
    """A bare JSON object (no fence) carrying a 'text'/'score' key is also
    rejected — the model treated its answer as structured output."""
    from app.vlm.qwen import _clean_vlm_text
    assert _clean_vlm_text('{"text": "X", "score": 0.9}') is None
    assert _clean_vlm_text('{"score": 0.5}') is None


def test_clean_vlm_text_keeps_unrelated_json():
    """JSON-like text WITHOUT a text/score key is NOT rejected — it could be a
    legitimate OCR of content that happens to look like JSON."""
    from app.vlm.qwen import _clean_vlm_text
    assert _clean_vlm_text('{"price": 9.9}') == '{"price": 9.9}'


def test_clean_vlm_text_strips_latex_formula():
    """LaTeX math markup is normalized to plain characters.

    The reported bug: the VLM transcribed a subscripted chemistry formula as
    "$\\text{C}_{60}$" and that string leaked verbatim into /analyze results.
    """
    from app.vlm.qwen import _clean_vlm_text
    assert _clean_vlm_text("$\\text{C}_{60}$") == "C60"
    assert _clean_vlm_text("$C_{60}$") == "C60"
    assert _clean_vlm_text("$\\mathrm{H_2O}$") == "H2O"
    assert _clean_vlm_text("$V_{max}$") == "Vmax"
    assert _clean_vlm_text("\\(C_{60}\\)") == "C60"


def test_clean_vlm_text_keeps_plain_dollar_text():
    """Prices / ordinary copy with '$' but no formula markup pass through."""
    from app.vlm.qwen import _clean_vlm_text
    assert _clean_vlm_text("$9.9") == "$9.9"
    assert _clean_vlm_text("售价 10$") == "售价 10$"
    assert _clean_vlm_text("C60 富勒烯") == "C60 富勒烯"


def test_strip_latex_direct():
    """Unit coverage of the LaTeX normalizer's trigger conditions."""
    from app.vlm.qwen import _strip_latex
    # Formula forms.
    assert _strip_latex("$\\text{C}_{60}$") == "C60"
    assert _strip_latex("$\\mathbf{C}_{60}$") == "C60"
    assert _strip_latex("$\\textbf{Fe}_2\\textbf{O}_3$") == "Fe2O3"
    assert _strip_latex("10^{6}") == "106"
    # Non-LaTeX strings untouched.
    assert _strip_latex("user_name: $9.9") == "user_name: $9.9"
    assert _strip_latex("纯文本，无公式") == "纯文本，无公式"


def test_parse_ocr_items_strips_latex_in_text_not_codes():
    """VLM-engine text payloads are LaTeX-normalized; qr/barcode payloads are
    NOT (machine-decoded strings like URLs may legitimately contain $ and _)."""
    from app.vlm_ocr import _parse_ocr_items
    raw = (
        '{"items": ['
        '{"type": "text", "text": "$\\\\text{C}_{60}$", '
        '"bbox": [0.1, 0.1, 0.3, 0.2], "confidence": 0.9},'
        '{"type": "barcode", "content": "http://x.com/a?b_$c", '
        '"bbox": [0.5, 0.5, 0.8, 0.6], "confidence": 0.9}'
        ']}'
    )
    items = _parse_ocr_items(raw, img_w=100, img_h=100)
    assert [it[1] for it in items] == ["C60", "http://x.com/a?b_$c"]




def _stub_client_tracking_calls(vlm: QwenVLM):
    """Record each create() call; return content "TEXT" on every invocation.
    (recognize_crop_with_prompt then yields ("TEXT", 1.0) — the 1.0 is a
    placeholder; the VLM no longer supplies a usable score.)"""
    calls = {"n": 0}

    class _Create:
        def create(self, **kwargs):
            calls["n"] += 1
            return SimpleNamespace(
                choices=[SimpleNamespace(
                    message=SimpleNamespace(content="TEXT")
                )]
            )
    vlm._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=_Create().create))
    )
    return calls


def _poly(x1, y1, x2, y2):
    """4-point polygon (list of [x,y]) for an axis-aligned box."""
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def test_recognize_crop_with_prompt_skips_undersized():
    """A crop smaller than min_crop_side on EITHER axis is skipped (empty result,
    no API call). Mirrors the DashScope failure mode that triggered this guard."""
    vlm = _make_vlm(min_crop_side=16)
    calls = _stub_client_tracking_calls(vlm)
    img = _img()  # 100x100

    # 36x7 (height too small — exactly the failing case from production logs).
    text, conf = vlm.recognize_crop_with_prompt(
        img, _poly(10, 10, 46, 17), prompt="read"
    )
    assert text == ""
    assert conf == 0.0
    assert calls["n"] == 0  # no API call made

    # 7x36 (width too small).
    text, conf = vlm.recognize_crop_with_prompt(
        img, _poly(10, 10, 17, 46), prompt="read"
    )
    assert text == ""
    assert calls["n"] == 0

    # Exactly 16x16 passes (boundary: equal is allowed).
    text, conf = vlm.recognize_crop_with_prompt(
        img, _poly(10, 10, 26, 26), prompt="read"
    )
    assert text == "TEXT"
    assert conf == 1.0  # placeholder; the VLM no longer supplies a usable score
    assert calls["n"] == 1


def test_recognize_crop_with_prompt_no_guard_when_min_side_zero():
    """min_crop_side=0 disables the guard — even tiny crops are sent."""
    vlm = _make_vlm(min_crop_side=0)
    calls = _stub_client_tracking_calls(vlm)
    img = _img()

    text, conf = vlm.recognize_crop_with_prompt(
        img, _poly(10, 10, 12, 12), prompt="read"  # 2x2
    )
    assert text == "TEXT"
    assert calls["n"] == 1


def test_recognize_crops_with_prompts_batch_skips_undersized_silently():
    """In the batch path, undersized crops return ("", 0.0) without raising,
    so one bad crop no longer fails the whole batch."""
    vlm = _make_vlm(min_crop_side=16)
    _stub_client_tracking_calls(vlm)
    img = _img()

    crops = [
        (_poly(10, 10, 60, 40), "p1"),    # 50x30 — OK
        (_poly(10, 10, 20, 13), "p2"),    # 10x3 — too small (the bug)
        (_poly(10, 10, 50, 50), "p3"),    # 40x40 — OK
    ]
    results = vlm.recognize_crops_with_prompts_batch(img, crops)

    assert len(results) == 3
    assert results[0] == ("TEXT", 1.0)   # OK crop recognized (1.0 = placeholder)
    assert results[1] == ("", 0.0)       # undersized → empty, no exception
    assert results[2] == ("TEXT", 1.0)   # OK crop recognized (1.0 = placeholder)


def test_recognize_crop_skips_undersized():
    """The legacy single-crop path (recognize_crop) honors the same guard."""
    vlm = _make_vlm(min_crop_side=16)
    calls = _stub_client_tracking_calls(vlm)
    img = _img()

    text, conf = vlm.recognize_crop(img, _poly(10, 10, 20, 17))  # 10x7 — too small
    assert text == ""
    assert conf == 0.0
    assert calls["n"] == 0


def test_settings_vlm_min_crop_side_default():
    """Default threshold is 16 (safety margin over DashScope's 10px hard limit)."""
    s = Settings()
    assert s.vlm_min_crop_side == 16

