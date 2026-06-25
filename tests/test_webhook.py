"""Tests for the outbound webhook notifier (``callback_url`` on ``/analyze``).

Two layers, following the repo convention (see ``test_fetch.py``):
1. ``app.webhook`` units — payload shape, HMAC signing determinism, and that
   ``deliver`` never raises (swallows network/non-2xx errors).
2. ``/analyze`` integration — sync path (small image) and async path (large
   image) both fire a signed callback when ``callback_url`` is set, and the
   receiver can pull the full result via ``/tasks/{id}``.

Outbound HTTP is captured via ``httpx.MockTransport`` patched onto
``httpx.Client`` (the sync client ``webhook.deliver`` uses). Delivery runs on a
background thread pool, so tests block on a ``threading.Event`` the mock handler
sets once it sees the POST.
"""
from __future__ import annotations

import hashlib
import hmac
import io
import json
import threading
import time

import httpx
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from app import main as main_mod
from app.config import Settings
from app.webhook import build_payload, deliver, sign


# ---------------------------------------------------------------------------
# unit tests — build_payload / sign / deliver
# ---------------------------------------------------------------------------


def test_build_payload_done_has_event_and_timestamp():
    p = build_payload("task-1", "done", biz_id="order-7", now=1_750_000_000.0)
    assert p["event"] == "analyze.completed"
    assert p["status"] == "done"
    assert p["task_id"] == "task-1"
    assert p["biz_id"] == "order-7"
    assert p["timestamp"].endswith("Z")  # ISO-8601 UTC marker
    assert "error" not in p


def test_build_payload_failed_carries_error():
    p = build_payload("task-2", "error", error="Boom: x")
    assert p["event"] == "analyze.failed"
    assert p["status"] == "error"
    assert p["error"] == "Boom: x"
    assert "biz_id" not in p  # omitted when absent (fixed schema)


def test_sign_matches_independent_recomputation():
    secret = "topsecret"
    body = json.dumps(build_payload("t", "done", now=0)).encode()
    sig = sign(body, secret)
    assert sig.startswith("sha256=")
    # The receiver recomputes over the raw body and compares the hex digest.
    expected_hex = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    assert sig == f"sha256={expected_hex}"


def test_sign_changes_when_body_changes():
    secret = "k"
    b1 = json.dumps(build_payload("t1", "done", now=0)).encode()
    b2 = json.dumps(build_payload("t2", "done", now=0)).encode()
    assert sign(b1, secret) != sign(b2, secret)


def test_deliver_never_raises_on_network_error():
    """A dead/invalid endpoint is logged, not raised — OCR must not be affected."""
    # ``deliver`` opens a real client; pointing it at an unroutable host with a
    # tiny timeout guarantees a connection error, which it must swallow.
    deliver(
        "task-x", "done", None,
        "http://127.0.0.1:1/not-listening",  # nothing bound on :1
        None, timeout=0.5,
    )
    # no exception = pass


# ---------------------------------------------------------------------------
# integration helpers
# ---------------------------------------------------------------------------


def _png_bytes(size=(8, 8)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (255, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


def _capture_post(monkeypatch, received: dict, event: threading.Event):
    """Patch ``httpx.Client`` so webhook POSTs hit a MockTransport that records
    the request body + headers, then signals ``event``.

    ``webhook.deliver`` builds ``httpx.Client(...)`` internally; we replace the
    class so every instance it creates is wired to our mock transport (same
    trick ``test_fetch.py`` uses for the async client).
    """
    transport = httpx.MockTransport(lambda req: httpx.Response(200, json={"ok": True}))
    real_client = httpx.Client

    def _factory(*args, **kwargs):
        kwargs["transport"] = transport
        client = real_client(*args, **kwargs)
        real_post = client.post

        def _post(url, *, content=None, headers=None, **kw):
            received["url"] = str(url)
            received["content"] = content
            received["headers"] = dict(headers or {})
            event.set()
            return real_post(url, content=content, headers=headers, **kw)

        client.post = _post  # type: ignore[method-assign]
        return client

    monkeypatch.setattr(httpx, "Client", _factory)


def _stub_pipeline_run(monkeypatch, run_impl):
    """Replace ``Pipeline.run`` with ``run_impl`` so no OCR model loads.

    Builds the lazily-created pipeline singleton first, then patches its bound
    method — the endpoint resolves the pipeline via ``_get_pipeline()`` which
    returns the same cached instance.
    """
    pipeline = main_mod._get_pipeline()
    monkeypatch.setattr(pipeline, "run", run_impl)


def _ok_response(width: int = 8, height: int = 8):
    """A minimal valid AnalyzeResponse for the stub to return (no items)."""
    from app.schemas import AnalyzeResponse, ImageMeta

    return AnalyzeResponse(image_meta=ImageMeta(width=width, height=height), items=[])


def _settings(**overrides) -> Settings:
    base = Settings()
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


def _wait(event: threading.Event, timeout: float = 5.0):
    """Block until the background webhook delivery completes, or fail loudly."""
    if not event.wait(timeout):
        pytest.fail("webhook was not delivered within timeout")


# ---------------------------------------------------------------------------
# integration — sync path (small image)
# ---------------------------------------------------------------------------


def test_sync_path_with_callback_fires_signed_webhook(monkeypatch):
    """Small image: callback set → result published under task_id + webhook POSTed."""
    received: dict = {}
    event = threading.Event()
    _capture_post(monkeypatch, received, event)
    monkeypatch.setattr(main_mod, "_settings", lambda: _settings())
    _stub_pipeline_run(
        monkeypatch,
        lambda image_data, annotate=False, options=None: _ok_response(),
    )

    c = TestClient(main_mod.app)
    r = c.post(
        "/analyze",
        files={"file": ("x.png", _png_bytes(), "image/png")},
        data={
            "callback_url": "http://biz.test/hook",
            "callback_secret": "s3cr3t",
            "biz_id": "order-42",
        },
    )
    assert r.status_code == 200
    task_id = r.json()["task_id"]
    assert task_id  # backfilled for the callback contract

    _wait(event)

    # The receiver can fetch the full result via the task id (same as async).
    body = c.get(f"/tasks/{task_id}").json()
    assert body["status"] == "done"

    # The webhook POST carries the signed status ping.
    assert received["url"] == "http://biz.test/hook"
    payload = json.loads(received["content"])
    assert payload["event"] == "analyze.completed"
    assert payload["task_id"] == task_id
    assert payload["biz_id"] == "order-42"
    assert payload["status"] == "done"
    assert "error" not in payload

    # Signature header present and valid over the exact bytes we sent.
    assert "X-Webhook-Signature" in received["headers"]
    sig = received["headers"]["X-Webhook-Signature"]
    expected = "sha256=" + hmac.new(b"s3cr3t", received["content"], hashlib.sha256).hexdigest()
    assert sig == expected


def test_sync_path_without_callback_is_unchanged(monkeypatch):
    """No callback_url → no task_id backfill, no webhook (backward compatible)."""
    event = threading.Event()
    received: dict = {}
    # If a webhook were fired, the handler would set the event; assert it never does.
    _capture_post(monkeypatch, received, event)
    monkeypatch.setattr(main_mod, "_settings", lambda: _settings())
    _stub_pipeline_run(
        monkeypatch,
        lambda image_data, annotate=False, options=None: _ok_response(),
    )

    c = TestClient(main_mod.app)
    r = c.post("/analyze", files={"file": ("x.png", _png_bytes(), "image/png")})
    assert r.status_code == 200
    assert r.json()["task_id"] is None  # no backfill without a callback
    assert not event.is_set()


def test_no_secret_means_no_signature_header(monkeypatch):
    received: dict = {}
    event = threading.Event()
    _capture_post(monkeypatch, received, event)
    monkeypatch.setattr(main_mod, "_settings", lambda: _settings())
    _stub_pipeline_run(
        monkeypatch,
        lambda image_data, annotate=False, options=None: _ok_response(),
    )

    c = TestClient(main_mod.app)
    r = c.post(
        "/analyze",
        files={"file": ("x.png", _png_bytes(), "image/png")},
        data={"callback_url": "http://biz.test/hook"},  # no secret
    )
    assert r.status_code == 200
    _wait(event)
    assert "X-Webhook-Signature" not in received["headers"]


def test_invalid_callback_url_returns_400(monkeypatch):
    """A non-http(s) callback URL is rejected before any OCR work."""
    monkeypatch.setattr(main_mod, "_settings", lambda: _settings())
    c = TestClient(main_mod.app)
    r = c.post(
        "/analyze",
        files={"file": ("x.png", _png_bytes(), "image/png")},
        data={"callback_url": "ftp://nope.test/x"},
    )
    assert r.status_code == 400
    assert "callback_url" in r.json()["detail"]


# ---------------------------------------------------------------------------
# integration — async path (large image)
# ---------------------------------------------------------------------------


def test_async_path_with_callback_fires_webhook(monkeypatch):
    """Large image → async → on 'done' a signed webhook is POSTed."""
    received: dict = {}
    event = threading.Event()
    _capture_post(monkeypatch, received, event)
    # Force the async path with a tiny threshold + an image bigger than it.
    monkeypatch.setattr(main_mod, "_settings", lambda: _settings(large_image_threshold=4))
    _stub_pipeline_run(
        monkeypatch,
        lambda image_data, annotate=False, options=None: _ok_response(),
    )

    c = TestClient(main_mod.app)
    r = c.post(
        "/analyze",
        files={"file": ("big.png", _png_bytes(), "image/png")},
        data={
            "callback_url": "http://biz.test/hook",
            "callback_secret": "k",
            "biz_id": "job-9",
        },
    )
    assert r.status_code == 202
    task_id = r.json()["task_id"]

    # The worker thread flips the task to done; poll until then.
    deadline = time.time() + 5.0
    while time.time() < deadline:
        st = c.get(f"/tasks/{task_id}").json().get("status")
        if st == "done":
            break
        time.sleep(0.02)
    else:
        pytest.fail("async task never reached 'done'")

    _wait(event)
    payload = json.loads(received["content"])
    assert payload["event"] == "analyze.completed"
    assert payload["task_id"] == task_id
    assert payload["status"] == "done"
    sig = received["headers"]["X-Webhook-Signature"]
    expected = "sha256=" + hmac.new(b"k", received["content"], hashlib.sha256).hexdigest()
    assert sig == expected


def test_async_path_failure_fires_failed_webhook(monkeypatch):
    """When the OCR worker throws, the callback reports 'error' (no silent hang)."""
    received: dict = {}
    event = threading.Event()
    _capture_post(monkeypatch, received, event)
    monkeypatch.setattr(main_mod, "_settings", lambda: _settings(large_image_threshold=4))

    def _boom(image_data, annotate=False, options=None):
        raise RuntimeError("OCR exploded")

    _stub_pipeline_run(monkeypatch, _boom)

    c = TestClient(main_mod.app)
    r = c.post(
        "/analyze",
        files={"file": ("bad.png", _png_bytes(), "image/png")},
        data={"callback_url": "http://biz.test/hook", "callback_secret": "k"},
    )
    assert r.status_code == 202
    task_id = r.json()["task_id"]

    deadline = time.time() + 5.0
    while time.time() < deadline:
        st = c.get(f"/tasks/{task_id}").json().get("status")
        if st == "error":
            break
        time.sleep(0.02)
    else:
        pytest.fail("async task never reached 'error'")

    _wait(event)
    payload = json.loads(received["content"])
    assert payload["event"] == "analyze.failed"
    assert payload["status"] == "error"
    assert "OCR exploded" in payload["error"]
    # And the task itself records the failure for /tasks polling.
    assert c.get(f"/tasks/{task_id}").json()["status"] == "error"
