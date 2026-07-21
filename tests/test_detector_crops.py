"""Tests for the detector-only box recovery (unrecognized crops).

The recognizer (PP-OCRv6 medium_rec) has no Hangul in its dictionary, so Korean
text boxes get a near-zero rec score and are filtered out by the pipeline. Before
this fix, those boxes vanished entirely because we only read `rec_polys`. Now we
also read `dt_polys` (the detector's full set) and emit the unmatched boxes as
`recognized=False` items, optionally with a base64 PNG crop.

These tests run WITHOUT paddle installed: the OCREngine's `predict` is monkeypatched
to return a fake result dict with controllable `dt_polys` / `rec_polys`.
"""
from __future__ import annotations

import base64
import io

import numpy as np
import pytest
from PIL import Image

from app.config import Settings
from app.ocr.detector import OCREngine, TextDetection, _encode_crop


# ---------------------------------------------------------------------------
# Fake PaddleOCR result: dt_polys has MORE boxes than rec_polys (the extras are
# the ones the recognizer dropped — e.g. Korean).
# ---------------------------------------------------------------------------


def _quad(x1, y1, x2, y2):
    """A 4-point polygon as a list of [x, y] pairs (list, not np)."""
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


class _FakeOCR:
    """Stand-in for PaddleOCR: returns a canned result from predict()."""

    def __init__(self, result):
        self._result = result

    def predict(self, arr, **kwargs):
        return [self._result]


def _make_engine(monkeypatch, result, *, emit_crops: bool):
    """Build an OCREngine whose model is faked and predict() returns `result`."""
    s = Settings()  # defaults; ocr_emit_crops=True by default
    monkeypatch.setattr(s, "ocr_emit_crops", emit_crops)
    engine = OCREngine(s)
    engine._ocr = _FakeOCR(result)  # bypass _ensure_loaded / paddle import
    return engine


def _result(rec_polys, rec_texts, rec_scores, dt_polys):
    return {
        "dt_polys": dt_polys,
        "rec_polys": rec_polys,
        "rec_texts": rec_texts,
        "rec_scores": rec_scores,
    }


# A 100x100 white tile (content doesn't matter; crop just needs to be non-empty).
_TILE = np.full((100, 200, 3), 255, dtype="uint8")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_detect_returns_unrecognized_for_filtered_polys(monkeypatch):
    """A dt_poly with no matching rec_poly → emitted as recognized=False."""
    result = _result(
        rec_polys=[_quad(10, 10, 60, 40)],          # one recognized box
        rec_texts=["CAUTION"],
        rec_scores=[0.95],
        dt_polys=[_quad(10, 10, 60, 40), _quad(70, 10, 120, 40)],  # + one dropped
    )
    engine = _make_engine(monkeypatch, result, emit_crops=False)
    dets = engine.detect_and_recognize(_TILE, granularity="line")

    assert len(dets) == 2
    rec = [d for d in dets if d.recognized]
    unrec = [d for d in dets if not d.recognized]
    assert len(rec) == 1
    assert rec[0].text == "CAUTION"
    assert len(unrec) == 1
    assert unrec[0].recognized is False
    assert unrec[0].text == ""
    assert unrec[0].confidence == 0.0
    # emit_crops=False → no crop
    assert unrec[0].crop_b64 is None


def test_detect_emits_crop_when_enabled(monkeypatch):
    """emit_crops=True → unrecognized box carries a base64 PNG crop."""
    result = _result(
        rec_polys=[],
        rec_texts=[],
        rec_scores=[],
        dt_polys=[_quad(10, 10, 60, 40)],
    )
    engine = _make_engine(monkeypatch, result, emit_crops=True)
    dets = engine.detect_and_recognize(_TILE, granularity="line")

    assert len(dets) == 1
    d = dets[0]
    assert d.recognized is False
    assert d.crop_b64 is not None
    # Decodes to a valid PNG.
    raw = base64.b64decode(d.crop_b64)
    img = Image.open(io.BytesIO(raw))
    assert img.format == "PNG"
    # Crop covers the bbox [10,10,60,40] → 50x30.
    assert img.size == (50, 30)


def test_recognized_items_have_no_crop(monkeypatch):
    """Recognized boxes never carry a crop, even when emit_crops is on."""
    result = _result(
        rec_polys=[_quad(10, 10, 60, 40)],
        rec_texts=["CAUTION"],
        rec_scores=[0.95],
        dt_polys=[_quad(10, 10, 60, 40)],
    )
    engine = _make_engine(monkeypatch, result, emit_crops=True)
    dets = engine.detect_and_recognize(_TILE, granularity="line")

    assert len(dets) == 1
    assert dets[0].recognized is True
    assert dets[0].crop_b64 is None


def test_detect_matches_dt_to_rec_by_iou(monkeypatch):
    """A dt_poly that overlaps a rec_poly (IoU ≥ 0.5) is NOT double-counted."""
    # dt and rec cover the same region (slightly different coords) → same box.
    result = _result(
        rec_polys=[_quad(10, 10, 60, 40)],
        rec_texts=["CAUTION"],
        rec_scores=[0.95],
        dt_polys=[_quad(12, 11, 62, 41)],  # ~same box, high IoU
    )
    engine = _make_engine(monkeypatch, result, emit_crops=True)
    dets = engine.detect_and_recognize(_TILE, granularity="line")

    # Only the recognized one; the dt box matched it, no duplicate.
    assert len(dets) == 1
    assert dets[0].recognized is True


def test_encode_crop_returns_none_for_degenerate():
    """Empty/zero-size polygon → None, never raises."""
    # 1x1 crop → too small.
    assert _encode_crop(_TILE, _quad(5, 5, 5, 5)) is None
    # Out-of-bounds polygon → empty crop.
    assert _encode_crop(_TILE, [[500, 500], [600, 500], [600, 600], [500, 600]]) is None


def test_no_unrecognized_in_paragraph_mode(monkeypatch):
    """Paragraph mode skips recovery (unrecognized lines handled downstream)."""
    result = _result(
        rec_polys=[_quad(10, 10, 60, 40)],
        rec_texts=["CAUTION"],
        rec_scores=[0.95],
        dt_polys=[_quad(10, 10, 60, 40), _quad(70, 10, 120, 40)],
    )
    engine = _make_engine(monkeypatch, result, emit_crops=True)
    dets = engine.detect_and_recognize(_TILE, granularity="paragraph")

    # All returned dets are recognized paragraph blocks (recovery skipped).
    assert all(d.recognized for d in dets)


# ---------------------------------------------------------------------------
# OCR stats accounting (per-tile predict timing + det/rec box counts)
# ---------------------------------------------------------------------------


def test_stats_track_predict_calls_and_box_counts(monkeypatch):
    """detect_and_recognize() accumulates predict_calls, box counts, and time."""
    result = _result(
        rec_polys=[_quad(10, 10, 60, 40), _quad(70, 10, 120, 40)],
        rec_texts=["A", "B"],
        rec_scores=[0.95, 0.88],
        dt_polys=[_quad(10, 10, 60, 40), _quad(70, 10, 120, 40), _quad(0, 0, 9, 9)],
    )
    engine = _make_engine(monkeypatch, result, emit_crops=False)
    assert engine.ocr_stats == {
        "predict_calls": 0, "boxes_detected": 0,
        "boxes_recognized": 0, "t_predict": 0.0,
    }

    engine.detect_and_recognize(_TILE, granularity="line")

    s = engine.ocr_stats
    assert s["predict_calls"] == 1
    assert s["boxes_detected"] == 3   # len(dt_polys)
    assert s["boxes_recognized"] == 2  # len(rec_polys)
    assert s["t_predict"] >= 0.0       # perf_counter delta (>= 0)

    # Second call accumulates.
    engine.detect_and_recognize(_TILE, granularity="line")
    s = engine.ocr_stats
    assert s["predict_calls"] == 2
    assert s["boxes_detected"] == 6
    assert s["boxes_recognized"] == 4


def test_stats_reset_clears_counters(monkeypatch):
    """reset_stats() zeroes every field so the next request starts clean."""
    result = _result(
        rec_polys=[_quad(10, 10, 60, 40)],
        rec_texts=["A"],
        rec_scores=[0.9],
        dt_polys=[_quad(10, 10, 60, 40)],
    )
    engine = _make_engine(monkeypatch, result, emit_crops=False)
    engine.detect_and_recognize(_TILE, granularity="line")
    assert engine.ocr_stats["predict_calls"] == 1

    engine.reset_stats()
    assert engine.ocr_stats == {
        "predict_calls": 0, "boxes_detected": 0,
        "boxes_recognized": 0, "t_predict": 0.0,
    }


def test_stats_empty_result_still_accounts_for_call(monkeypatch):
    """A predict() returning no result still counts the call and its time."""
    engine = _make_engine(monkeypatch, None, emit_crops=False)
    engine._ocr = type(engine._ocr)(None)  # FakeOCR with None result
    dets = engine.detect_and_recognize(_TILE, granularity="line")

    assert dets == []
    s = engine.ocr_stats
    assert s["predict_calls"] == 1
    assert s["boxes_detected"] == 0
    assert s["boxes_recognized"] == 0


# ---------------------------------------------------------------------------
# _ensure_loaded forwards CPU runtime kwargs to PaddleOCR(...)
# ---------------------------------------------------------------------------


class _PaddleOCRSpy:
    """Captures the kwargs OCREngine passes to PaddleOCR().

    Installed in place of `paddleocr.PaddleOCR` so we can assert the runtime
    tuning knobs (cpu_threads / text_recognition_batch_size / enable_mkldnn)
    are forwarded correctly without loading any real model.
    """

    def __init__(self, *args, **kwargs):
        self.kwargs = kwargs

    def predict(self, arr, **kwargs):
        return []


def _install_paddleocr_spy(monkeypatch):
    """Patch `from paddleocr import PaddleOCR` inside detector._ensure_loaded.

    Returns the spy class; after `_ensure_loaded()` runs, `spy.kwargs` holds the
    kwargs the engine forwarded.
    """
    import sys
    spy = _PaddleOCRSpy
    fake_module = type("M", (), {"PaddleOCR": spy})
    monkeypatch.setitem(sys.modules, "paddleocr", fake_module)
    return spy


def test_ensure_loaded_forwards_cpu_runtime_overrides(monkeypatch):
    """When cpu_threads/rec_batch_size are set, they reach PaddleOCR()."""
    spy = _install_paddleocr_spy(monkeypatch)
    s = Settings()
    monkeypatch.setattr(s, "ocr_cpu_threads", 20)
    monkeypatch.setattr(s, "ocr_rec_batch_size", 50)
    monkeypatch.setattr(s, "ocr_enable_mkldnn", False)

    engine = OCREngine(s)
    engine._ensure_loaded()

    kws = engine._ocr.kwargs
    assert kws["cpu_threads"] == 20
    assert kws["text_recognition_batch_size"] == 50
    assert kws["enable_mkldnn"] is False


def test_ensure_loaded_defaults_omit_cpu_threads_and_rec_batch(monkeypatch):
    """At default settings, only enable_mkldnn is forwarded (= True).

    cpu_threads=0 and rec_batch_size=None must NOT be passed so PaddleOCR's own
    defaults kick in (preserves the pre-tuning behavior).
    """
    spy = _install_paddleocr_spy(monkeypatch)
    s = Settings()  # all defaults

    engine = OCREngine(s)
    engine._ensure_loaded()

    kws = engine._ocr.kwargs
    assert "cpu_threads" not in kws
    assert "text_recognition_batch_size" not in kws
    # mkldnn is always forwarded (even at default True) — turning it off is a
    # documented debugging knob, so it's part of the contract.
    assert kws["enable_mkldnn"] is True
