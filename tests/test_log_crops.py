"""Tests for the /logs VLM crop thumbnails feature.

The pipeline stores a JPEG thumbnail of every crop sent to the VLM fallback
(aligned with the fallback_crops detail list); log_buffer keeps those images
in a byte-budgeted side store keyed by a stable record id; GET
/logs/crop/{rec_id}/{idx} serves them to the /logs UI, which shows one
thumbnail per crop row.
"""
from __future__ import annotations

import numpy as np

from app.config import Settings
from app.log_buffer import (
    LogRecord,
    clear as lb_clear,
    log_buffer_crop_image,
    log_buffer_record,
    log_buffer_snapshot,
)
from app.pipeline import Pipeline
from app.schemas import Item


JPEG_START = b"\xff\xd8"
JPEG_END = b"\xff\xd9"


def _img(h: int = 100, w: int = 100) -> np.ndarray:
    """An RGB test image."""
    rng = np.random.default_rng(7)
    return rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)


def _items() -> list[Item]:
    """Two low-confidence suspects (sent to the VLM) + one confident box."""
    return [
        Item(id="t1", type="text", text="?",
             polygon=[[10, 10], [30, 10], [30, 30], [10, 30]],
             bbox=[10, 10, 30, 30], confidence=0.70, source="paddleocr"),
        Item(id="t2", type="text", text="ok",
             polygon=[[40, 40], [60, 40], [60, 60], [40, 60]],
             bbox=[40, 40, 60, 60], confidence=0.99, source="paddleocr"),
        Item(id="t3", type="text", text="?",
             polygon=[[70, 70], [90, 70], [90, 90], [70, 90]],
             bbox=[70, 70, 90, 90], confidence=0.80, source="paddleocr"),
    ]


def _pipe_with_fake_vlm(monkeypatch) -> Pipeline:
    """Pipeline with the VLM fallback force-enabled and a stubbed VLM."""
    pipe = Pipeline(Settings())
    pipe.settings = pipe.settings.model_copy(update={
        "vlm_enabled": True,
        "vlm_ocr_fallback_enabled": True,
        "rec_confidence_fallback": 0.99,
        "rec_confidence_drop": 0.60,
        "min_keep_confidence": 0.60,
    })

    class _FakeVLM:
        def recognize_crops_with_prompts_batch(self, image, crops):
            return [("HELLO", 0.8) for _ in crops]

    monkeypatch.setattr(pipe, "_get_vlm", lambda: _FakeVLM())
    return pipe


# ---------------------------------------------------------------------------
# Pipeline: thumbnails ride alongside fallback_crops
# ---------------------------------------------------------------------------


def test_fallback_stats_carry_aligned_crop_images(monkeypatch):
    """stats['crop_images'] is aligned 1:1 with stats['crops'] — one JPEG per
    crop row, in the same order, so the UI can map row i → image i."""
    pipe = _pipe_with_fake_vlm(monkeypatch)
    _out, n_crops, stats = pipe._maybe_vlm_fallback(_img(), _items())

    assert n_crops == 2
    crops, images = stats["crops"], stats["crop_images"]
    assert len(crops) == len(images) == 2
    for im in images:
        assert isinstance(im, bytes)
        assert im.startswith(JPEG_START) and im.endswith(JPEG_END)


def test_crop_thumbnail_small_box_kept_original_size(monkeypatch):
    """A crop smaller than the thumb cap is stored 1:1 (no upscale)."""
    from io import BytesIO

    from PIL import Image

    pipe = _pipe_with_fake_vlm(monkeypatch)
    thumb = pipe._crop_thumbnail(_img(), [10, 10, 30, 30])  # 20x20 crop
    assert thumb is not None
    pil = Image.open(BytesIO(thumb))
    assert pil.size == (20, 20)


def test_crop_thumbnail_large_box_downscaled(monkeypatch):
    """A crop larger than the thumb cap is downscaled to fit it."""
    from io import BytesIO

    from PIL import Image

    pipe = _pipe_with_fake_vlm(monkeypatch)
    thumb = pipe._crop_thumbnail(_img(600, 800), [0, 0, 800, 600])  # full frame
    assert thumb is not None
    pil = Image.open(BytesIO(thumb))
    assert max(pil.size) <= 200


def test_crop_thumbnail_degenerate_box_returns_none():
    """An empty/out-of-image box yields None (UI shows no thumbnail)."""
    pipe = Pipeline(Settings())
    assert pipe._crop_thumbnail(_img(), [50, 50, 50, 60]) is None
    assert pipe._crop_thumbnail(_img(), [500, 500, 600, 600]) is None


# ---------------------------------------------------------------------------
# log_buffer: id assignment + byte-budgeted image store
# ---------------------------------------------------------------------------


def _rec() -> LogRecord:
    return LogRecord(ts=1.0, task_id=None, src="x", engine="paddleocr")


def test_record_id_assigned_and_snapshot_exposes_it():
    lb_clear()
    rec = _rec()
    log_buffer_record(rec)
    assert rec.id > 0
    snap = log_buffer_snapshot()
    assert snap[-1]["id"] == rec.id


def test_crop_image_roundtrip_and_bounds():
    lb_clear()
    rec = _rec()
    log_buffer_record(rec, [b"img0", b"img1"])
    assert log_buffer_crop_image(rec.id, 0) == b"img0"
    assert log_buffer_crop_image(rec.id, 1) == b"img1"
    assert log_buffer_crop_image(rec.id, 2) is None      # index out of range
    assert log_buffer_crop_image(10 ** 9, 0) is None     # unknown record
    # Records registered without images serve no thumbnails.
    rec2 = _rec()
    log_buffer_record(rec2)
    assert log_buffer_crop_image(rec2.id, 0) is None


def test_non_bytes_entries_skipped():
    """None (encode failure) / junk entries are dropped at registration so the
    store only ever serves real bytes."""
    lb_clear()
    rec = _rec()
    log_buffer_record(rec, [b"ok", None, "junk"])
    assert log_buffer_crop_image(rec.id, 0) == b"ok"
    assert log_buffer_crop_image(rec.id, 1) is None


def test_byte_budget_evicts_oldest(monkeypatch):
    """Once the total byte budget is exceeded, the OLDEST record's images are
    evicted first (monotonic ids); newer lookups keep working."""
    from app import log_buffer

    monkeypatch.setattr(log_buffer, "IMAGES_MAX_BYTES", 10)
    lb_clear()
    r1, r2, r3 = _rec(), _rec(), _rec()
    log_buffer_record(r1, [b"A" * 6])
    log_buffer_record(r2, [b"B" * 6])   # budget 10 < 12 → r1 evicted
    log_buffer_record(r3, [b"C" * 6])   # budget 10 < 12 → r2 evicted
    assert log_buffer_crop_image(r1.id, 0) is None
    assert log_buffer_crop_image(r2.id, 0) is None
    assert log_buffer_crop_image(r3.id, 0) == b"C" * 6


def test_clear_drops_images():
    lb_clear()
    rec = _rec()
    log_buffer_record(rec, [b"img"])
    lb_clear()
    assert log_buffer_crop_image(rec.id, 0) is None
    assert log_buffer_snapshot() == []


# ---------------------------------------------------------------------------
# Endpoint: GET /logs/crop/{rec_id}/{idx}
# ---------------------------------------------------------------------------


def test_logs_crop_endpoint_serves_jpeg_and_404s():
    from fastapi import HTTPException
    from fastapi.testclient import TestClient

    from app.main import app

    lb_clear()
    rec = _rec()
    log_buffer_record(rec, [b"\xff\xd8xff\xd9"])

    client = TestClient(app)
    resp = client.get(f"/logs/crop/{rec.id}/0")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/jpeg"
    assert resp.content == b"\xff\xd8xff\xd9"

    # Unknown record / bad index → 404 (the UI's onerror hides the thumbnail).
    assert client.get(f"/logs/crop/{rec.id}/9").status_code == 404
    assert client.get("/logs/crop/999999/0").status_code == 404
