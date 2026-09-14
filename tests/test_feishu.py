"""Tests for the Feishu bot transport (``app/feishu.py``).

Two layers, following the repo convention (see ``test_webhook.py``):
1. ``app.feishu`` units — the 加签 signature algorithm (HMAC-SHA256 keyed on
   ``"<timestamp>\\n<secret>"`` over an EMPTY message, base64-encoded — NOT the
   body-signing scheme ``webhook.py`` uses), the ``msg_type=text`` payload
   shape, and that ``send_text`` never raises.
2. Delivery capture via ``httpx.MockTransport`` patched onto ``httpx.Client``
   (the sync client ``send_text`` constructs internally).

Slow/failure decision logic lives in ``tests/test_perf_alert.py``; this file
only proves the transport is correct and harmless.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import threading

import httpx
import pytest

from app.feishu import gen_sign, send_text


# ---------------------------------------------------------------------------
# unit tests — gen_sign / send_text
# ---------------------------------------------------------------------------


def test_gen_sign_matches_independent_recomputation():
    """Feishu's documented algorithm: key = f"{ts}\\n{secret}", empty message,
    base64 digest. Pins the scheme against an independent implementation."""
    ts, secret = 1_700_000_000, "test-secret"
    sig = gen_sign(ts, secret)
    expected = base64.b64encode(
        hmac.new(f"{ts}\n{secret}".encode("utf-8"), digestmod=hashlib.sha256).digest()
    ).decode("utf-8")
    assert sig == expected


def test_gen_sign_changes_with_timestamp_or_secret():
    assert gen_sign(1, "k") != gen_sign(2, "k")
    assert gen_sign(1, "k") != gen_sign(1, "k2")


def test_send_text_posts_msg_type_text(monkeypatch):
    received: dict = {}
    event = threading.Event()

    def _capture(monkeypatch, received, event):
        transport = httpx.MockTransport(
            lambda req: httpx.Response(200, json={"code": 0})
        )
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

    _capture(monkeypatch, received, event)
    ok = send_text("hello 观察者", "https://open.feishu.cn/hook/x")
    assert ok is True

    assert event.wait(5.0)
    assert received["url"] == "https://open.feishu.cn/hook/x"
    body = json.loads(received["content"])
    assert body["msg_type"] == "text"
    assert body["content"]["text"] == "hello 观察者"
    # No secret → no signing fields at all.
    assert "timestamp" not in body
    assert "sign" not in body


def test_send_text_with_secret_includes_matching_sign(monkeypatch):
    received: dict = {}
    event = threading.Event()
    transport = httpx.MockTransport(lambda req: httpx.Response(200, json={"code": 0}))
    real_client = httpx.Client

    def _factory(*args, **kwargs):
        kwargs["transport"] = transport
        client = real_client(*args, **kwargs)
        real_post = client.post

        def _post(url, *, content=None, headers=None, **kw):
            received["content"] = content
            event.set()
            return real_post(url, content=content, headers=headers, **kw)

        client.post = _post  # type: ignore[method-assign]
        return client

    monkeypatch.setattr(httpx, "Client", _factory)
    assert send_text("boom", "https://open.feishu.cn/hook/x", secret="s3cr3t")

    assert event.wait(5.0)
    body = json.loads(received["content"])
    # Body timestamp must be a STRING and the sign must verify against it.
    assert isinstance(body["timestamp"], str)
    ts = int(body["timestamp"])
    assert body["sign"] == gen_sign(ts, "s3cr3t")


def test_send_text_returns_false_on_http_error(monkeypatch):
    transport = httpx.MockTransport(lambda req: httpx.Response(500, json={}))
    real_client = httpx.Client

    def _factory(*args, **kwargs):
        kwargs["transport"] = transport
        return real_client(*args, **kwargs)

    monkeypatch.setattr(httpx, "Client", _factory)
    assert send_text("x", "https://open.feishu.cn/hook/x") is False


def test_send_text_never_raises_on_network_error():
    """Dead endpoint is logged, not raised — alerting must not break callers."""
    ok = send_text("x", "http://127.0.0.1:1/not-listening", timeout=0.5)
    assert ok is False


def test_send_text_noop_without_url():
    """Empty URL (alerting not configured) → False, no request attempted."""
    assert send_text("x", "") is False
