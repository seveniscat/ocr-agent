"""Tests for /analyze performance alerting (``app/perf_alert.py``).

Two layers, following the repo convention (see ``test_webhook.py``):
1. ``app.perf_alert`` units — the size-aware threshold formula, slow/error
   trigger decisions, per-kind cooldown suppression, and every off-switch.
2. ``/analyze`` integration — sync slow request and sync/async failures each
   push a Feishu text message; with no bot configured nothing is pushed.

The alerter decides but never sends (it returns the text; main.py submits it
to the webhook pool), so unit tests need no mocking at all. Integration tests
capture the outbound POST via ``httpx.MockTransport`` patched onto
``httpx.Client`` (same trick as ``test_webhook.py``) and block on a
``threading.Event``. Each integration test swaps in a fresh ``PerfAlerter``
so the module-level singleton's cooldown state can't leak between tests.
"""
from __future__ import annotations

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
from app.perf_alert import PerfAlerter, slow_threshold_seconds


# ---------------------------------------------------------------------------
# unit tests — threshold formula + decision logic
# ---------------------------------------------------------------------------


def _settings(**overrides) -> Settings:
    base = Settings()
    for k, v in overrides.items():
        setattr(base, k, v)
    return base


def _alerter(**overrides) -> tuple[PerfAlerter, Settings]:
    """A fresh alerter + alert-enabled settings (isolated cooldown state)."""
    settings = _settings(
        feishu_webhook_url="https://open.feishu.cn/hook/test",
        **overrides,
    )
    return PerfAlerter(), settings


def test_threshold_formula_is_base_plus_per_mp():
    settings = _settings(alert_slow_base_seconds=30.0, alert_slow_seconds_per_mp=5.0)
    assert slow_threshold_seconds(0.8, settings) == pytest.approx(34.0)
    assert slow_threshold_seconds(12.0, settings) == pytest.approx(90.0)
    assert slow_threshold_seconds(48.0, settings) == pytest.approx(270.0)


def test_slow_alert_fires_above_threshold_with_details():
    alerter, settings = _alerter()
    text = alerter.record_analyze(
        duration_s=95.0, status="ok", settings=settings,
        w=4000, h=3000, src="url=http://a/b.png", items=7,
        stats_sink={"t_preprocess": 1.2, "t_ocr": 80.1, "t_ocr_predict": 79.5,
                    "t_vlm": 4.0, "t_dedupe": 0.2, "t_annotate": 0.0},
    )
    assert text is not None
    assert "慢请求" in text
    assert "95.0s" in text and "90.0s" in text          # duration > threshold
    assert "4000x3000" in text and "url=http://a/b.png" in text
    assert "items=7" in text
    assert "分段:" in text and "ocr=80.1s" in text and "predict=79.5s" in text


def test_fast_request_no_alert():
    alerter, settings = _alerter()
    text = alerter.record_analyze(
        duration_s=10.0, status="ok", settings=settings, w=1000, h=800,
    )
    assert text is None


def test_slow_alerts_off_when_both_thresholds_zero():
    """base=per_mp=0 is the documented off switch, not 'alert on everything'."""
    alerter, settings = _alerter(
        alert_slow_base_seconds=0.0, alert_slow_seconds_per_mp=0.0,
    )
    text = alerter.record_analyze(
        duration_s=10_000.0, status="ok", settings=settings, w=4000, h=3000,
    )
    assert text is None


def test_async_path_labeled_by_task_id():
    # per_mp must be zeroed too: default 5s/MP x 48MP would put the threshold
    # at 240s and neither 5s duration would trip it.
    alerter, settings = _alerter(
        alert_slow_base_seconds=0.001,
        alert_slow_seconds_per_mp=0.0,
        alert_cooldown_seconds=0.0,
    )
    text = alerter.record_analyze(
        duration_s=5.0, status="ok", settings=settings,
        w=8000, h=6000, task_id="task-42",
    )
    assert text is not None
    assert "async" in text and "task=task-42" in text
    sync_text = alerter.record_analyze(
        duration_s=5.0, status="ok", settings=settings, w=8000, h=6000,
    )
    assert sync_text is not None and "sync" in sync_text


def test_error_alert_fires_even_when_slow_disabled():
    alerter, settings = _alerter(
        alert_slow_base_seconds=0.0, alert_slow_seconds_per_mp=0.0,
    )
    text = alerter.record_analyze(
        duration_s=1.0, status="error", settings=settings,
        error="RuntimeError: OCR exploded", w=100, h=100,
    )
    assert text is not None
    assert "失败" in text and "OCR exploded" in text


def test_error_alert_off_switch():
    alerter, settings = _alerter(alert_on_error=False)
    text = alerter.record_analyze(
        duration_s=1.0, status="error", settings=settings, error="boom",
    )
    assert text is None


def test_cooldown_suppresses_same_kind_only():
    alerter, settings = _alerter(alert_cooldown_seconds=300.0)
    slow = lambda: alerter.record_analyze(  # noqa: E731
        duration_s=95.0, status="ok", settings=settings, w=4000, h=3000,
    )
    assert slow() is not None          # first fires
    assert slow() is None              # second within cooldown → suppressed
    # The error kind has its own cooldown — a failure right after a slow
    # alert still goes out.
    err = alerter.record_analyze(
        duration_s=1.0, status="error", settings=settings, error="x",
    )
    assert err is not None


def test_zero_cooldown_never_suppresses():
    alerter, settings = _alerter(alert_cooldown_seconds=0.0)
    first = alerter.record_analyze(
        duration_s=95.0, status="ok", settings=settings, w=4000, h=3000,
    )
    second = alerter.record_analyze(
        duration_s=95.0, status="ok", settings=settings, w=4000, h=3000,
    )
    assert first is not None and second is not None


def test_no_webhook_url_disables_everything():
    settings = _settings(feishu_webhook_url="")
    alerter = PerfAlerter()
    assert alerter.record_analyze(
        duration_s=95.0, status="ok", settings=settings, w=4000, h=3000,
    ) is None
    assert alerter.record_analyze(
        duration_s=1.0, status="error", settings=settings, error="boom",
    ) is None


# ---------------------------------------------------------------------------
# integration helpers (same patterns as test_webhook.py)
# ---------------------------------------------------------------------------


def _png_bytes(size=(8, 8)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, (255, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


def _capture_post(monkeypatch, received: dict, event: threading.Event):
    """Route every httpx.Client POST feishu.send_text makes into ``received``."""
    transport = httpx.MockTransport(lambda req: httpx.Response(200, json={"code": 0}))
    real_client = httpx.Client

    def _factory(*args, **kwargs):
        kwargs["transport"] = transport
        client = real_client(*args, **kwargs)
        real_post = client.post

        def _post(url, *, content=None, headers=None, **kw):
            received["url"] = str(url)
            received["content"] = content
            event.set()
            return real_post(url, content=content, headers=headers, **kw)

        client.post = _post  # type: ignore[method-assign]
        return client

    monkeypatch.setattr(httpx, "Client", _factory)


def _stub_pipeline_run(monkeypatch, run_impl):
    pipeline = main_mod._get_pipeline()

    def _run_wrapper(image_data, annotate=False, options=None, **_kwargs):
        return run_impl(image_data, annotate=annotate, options=options)

    monkeypatch.setattr(pipeline, "run", _run_wrapper)


def _ok_response(width: int = 8, height: int = 8):
    from app.schemas import AnalyzeResponse, ImageMeta

    return AnalyzeResponse(image_meta=ImageMeta(width=width, height=height), items=[])


def _alert_settings(**overrides) -> Settings:
    """Bot configured + threshold so low any request trips it; cooldowns off."""
    return _settings(
        feishu_webhook_url="https://open.feishu.cn/hook/test",
        alert_slow_base_seconds=0.001,
        alert_slow_seconds_per_mp=0.0,
        alert_cooldown_seconds=0.0,
        **overrides,
    )


def _wait(event: threading.Event, timeout: float = 5.0):
    if not event.wait(timeout):
        pytest.fail("feishu alert was not delivered within timeout")


# ---------------------------------------------------------------------------
# integration — /analyze hooks
# ---------------------------------------------------------------------------


def test_sync_slow_request_pushes_feishu_alert(monkeypatch):
    received: dict = {}
    event = threading.Event()
    _capture_post(monkeypatch, received, event)
    monkeypatch.setattr(main_mod, "_settings", lambda: _alert_settings())
    monkeypatch.setattr(main_mod, "_alerter", PerfAlerter())

    def _slow_run(image_data, annotate=False, options=None):
        # The 1ms threshold is far below any real request, but a stubbed run
        # can finish in under a millisecond — sleep past it deterministically.
        time.sleep(0.05)
        return _ok_response()

    _stub_pipeline_run(monkeypatch, _slow_run)

    c = TestClient(main_mod.app)
    r = c.post("/analyze", files={"file": ("x.png", _png_bytes(), "image/png")})
    assert r.status_code == 200

    _wait(event)
    body = json.loads(received["content"])
    assert received["url"] == "https://open.feishu.cn/hook/test"
    assert body["msg_type"] == "text"
    assert "慢请求" in body["content"]["text"]


def test_sync_failure_pushes_feishu_error_alert(monkeypatch):
    received: dict = {}
    event = threading.Event()
    _capture_post(monkeypatch, received, event)
    monkeypatch.setattr(main_mod, "_settings", lambda: _alert_settings())
    monkeypatch.setattr(main_mod, "_alerter", PerfAlerter())

    def _boom(image_data, annotate=False, options=None):
        raise ValueError("OCR exploded")

    _stub_pipeline_run(monkeypatch, _boom)

    c = TestClient(main_mod.app)
    r = c.post("/analyze", files={"file": ("x.png", _png_bytes(), "image/png")})
    assert r.status_code == 503

    _wait(event)
    body = json.loads(received["content"])
    assert "失败" in body["content"]["text"]
    assert "OCR exploded" in body["content"]["text"]


def test_async_failure_pushes_feishu_error_alert(monkeypatch):
    received: dict = {}
    event = threading.Event()
    _capture_post(monkeypatch, received, event)
    monkeypatch.setattr(
        main_mod, "_settings",
        lambda: _alert_settings(large_image_threshold=4),
    )
    monkeypatch.setattr(main_mod, "_alerter", PerfAlerter())

    def _boom(image_data, annotate=False, options=None):
        raise RuntimeError("OCR exploded")

    _stub_pipeline_run(monkeypatch, _boom)

    c = TestClient(main_mod.app)
    r = c.post("/analyze", files={"file": ("big.png", _png_bytes(), "image/png")})
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
    body = json.loads(received["content"])
    text = body["content"]["text"]
    assert "失败" in text and "OCR exploded" in text
    assert "async" in text and f"task={task_id}" in text


def test_no_bot_configured_never_posts(monkeypatch):
    """No webhook URL → alerting fully off: no outbound POST, request intact."""
    event = threading.Event()
    received: dict = {}
    _capture_post(monkeypatch, received, event)
    monkeypatch.setattr(main_mod, "_settings", lambda: _settings())
    monkeypatch.setattr(main_mod, "_alerter", PerfAlerter())
    _stub_pipeline_run(
        monkeypatch,
        lambda image_data, annotate=False, options=None: _ok_response(),
    )

    c = TestClient(main_mod.app)
    r = c.post("/analyze", files={"file": ("x.png", _png_bytes(), "image/png")})
    assert r.status_code == 200
    assert not event.is_set()
