"""Package-detection task contract: batch_no / location / async_mode / batches API."""
from __future__ import annotations

import io
import time

import pytest
from PIL import Image
from fastapi.testclient import TestClient

from app import main as main_mod
from app.config import Settings
from app.webhook import build_payload


def _tiny_png_bytes(w: int = 64, h: int = 64) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), color=(240, 240, 240)).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def client(monkeypatch):
    # Avoid loading real OCR models; stub pipeline.run.
    class _FakePipeline:
        def run(self, data, **kwargs):
            from app.schemas import AnalyzeResponse, ImageMeta

            return AnalyzeResponse(
                image_meta=ImageMeta(width=64, height=64, tile_count=1),
                items=[],
            )

    monkeypatch.setattr(main_mod, "_get_pipeline", lambda: _FakePipeline())
    monkeypatch.setattr(main_mod, "get_settings", lambda: Settings())
    # Force "small image" threshold high so size alone wouldn't async.
    monkeypatch.setattr(
        main_mod,
        "_settings",
        lambda: Settings(large_image_threshold=99999, small_image_threshold=99999),
    )
    main_mod._tasks.clear()
    main_mod._batches.clear()
    return TestClient(main_mod.app)


def test_build_payload_includes_batch_and_location():
    p = build_payload(
        "t1",
        "done",
        biz_id="rec-9",
        batch_no="batch-A",
        location="front@0",
        now=0,
    )
    assert p["batch_no"] == "batch-A"
    assert p["location"] == "front@0"
    assert p["biz_id"] == "rec-9"


def test_batch_no_forces_async_and_composes_task_id(client):
    png = _tiny_png_bytes()
    resp = client.post(
        "/analyze",
        files={"file": ("t.png", png, "image/png")},
        data={
            "batch_no": "b-001",
            "location": "front@0",
            "biz_id": "42",
        },
    )
    assert resp.status_code == 202
    body = resp.json()
    assert body["task_id"] == "b-001::front@0"

    # Poll until done (fake pipeline is fast).
    deadline = time.time() + 5
    task = None
    while time.time() < deadline:
        task = client.get("/tasks/b-001::front@0").json()
        if task["status"] in ("done", "error"):
            break
        time.sleep(0.05)
    assert task is not None
    assert task["status"] == "done"
    assert task["batch_no"] == "b-001"
    assert task["location"] == "front@0"
    assert task["biz_id"] == "42"
    assert task["percent"] == 100


def test_explicit_task_id_and_batch_aggregate(client):
    png = _tiny_png_bytes()
    for loc, tid in (("front", "job-front"), ("back", "job-back")):
        r = client.post(
            "/analyze",
            files={"file": ("t.png", png, "image/png")},
            data={
                "task_id": tid,
                "batch_no": "batch-X",
                "location": loc,
                "async_mode": "true",
            },
        )
        assert r.status_code == 202
        assert r.json()["task_id"] == tid

    deadline = time.time() + 5
    batch = None
    while time.time() < deadline:
        br = client.get("/batches/batch-X")
        assert br.status_code == 200
        batch = br.json()
        if batch["status"] in ("done", "error", "partial") and batch["percent"] == 100:
            break
        time.sleep(0.05)

    assert batch is not None
    assert batch["total"] == 2
    assert batch["done"] == 2
    assert batch["status"] == "done"
    assert {t["task_id"] for t in batch["tasks"]} == {"job-front", "job-back"}


def test_duplicate_running_task_returns_409(client, monkeypatch):
    # Keep first job stuck in running so second collides.
    def _slow_run(self, data, **kwargs):
        time.sleep(2.0)
        from app.schemas import AnalyzeResponse, ImageMeta

        return AnalyzeResponse(
            image_meta=ImageMeta(width=64, height=64, tile_count=1),
            items=[],
        )

    class _SlowPipeline:
        run = _slow_run

    monkeypatch.setattr(main_mod, "_get_pipeline", lambda: _SlowPipeline())
    png = _tiny_png_bytes()
    r1 = client.post(
        "/analyze",
        files={"file": ("t.png", png, "image/png")},
        data={"task_id": "same-id", "async_mode": "1"},
    )
    assert r1.status_code == 202
    r2 = client.post(
        "/analyze",
        files={"file": ("t.png", png, "image/png")},
        data={"task_id": "same-id", "async_mode": "1"},
    )
    assert r2.status_code == 409


def test_batch_not_found():
    main_mod._tasks.clear()
    main_mod._batches.clear()
    client = TestClient(main_mod.app)
    r = client.get("/batches/no-such-batch")
    assert r.status_code == 404
