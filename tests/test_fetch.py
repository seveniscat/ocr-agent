"""Tests for URL image input (the ``url`` form field).

Two layers:
1. ``app.fetch.fetch_image`` — exercised via ``httpx.MockTransport`` so no real
   network I/O. Covers success, non-2xx, size-limit overrun, and the empty-body
   edge case.
2. The endpoint integration — ``_resolve_image`` precedence (file wins over
   url; neither → 400) and an end-to-end ``url`` POST through ``/panels/candidates``
   (pure geometry, no OCR/VLM model load — the cleanest endpoint to drive).

The repo's convention (see ``test_cut_lines.py``) is monkeypatch + ``TestClient``
with no real outbound calls; we follow that here.
"""
from __future__ import annotations

import asyncio
import io

import httpx
import pytest
from PIL import Image
from fastapi.testclient import TestClient

from app import main as main_mod
from app.config import Settings
from app.fetch import FetchError, fetch_image


# ---------------------------------------------------------------------------
# fetch_image — unit tests via httpx.MockTransport
# ---------------------------------------------------------------------------


def _png_bytes() -> bytes:
    """A tiny valid PNG, so a successful fetch returns decodable bytes."""
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), (255, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


def _client_with(handler):
    """Build a MockTransport-backed AsyncClient factory for monkeypatching.

    ``fetch_image`` constructs ``httpx.AsyncClient`` internally; we replace it
    so every client it builds is wired to our mock transport.
    """
    transport = httpx.MockTransport(handler)
    real_async_client = httpx.AsyncClient

    def _factory(*args, **kwargs):
        kwargs["transport"] = transport
        return real_async_client(*args, **kwargs)

    return _factory


def test_fetch_image_success(monkeypatch):
    """A 200 with a PNG body returns those bytes verbatim."""
    body = _png_bytes()
    monkeypatch.setattr(httpx, "AsyncClient", _client_with(lambda req: httpx.Response(200, content=body)))
    data = asyncio.run(fetch_image("http://x.test/a.png"))
    assert data == body


def test_fetch_image_non_2xx(monkeypatch):
    """A 404 surfaces as a FetchError mentioning the status code."""
    monkeypatch.setattr(httpx, "AsyncClient", _client_with(lambda req: httpx.Response(404, content=b"nope")))
    with pytest.raises(FetchError, match="404"):
        asyncio.run(fetch_image("http://x.test/missing.png"))


def test_fetch_image_size_limit(monkeypatch):
    """Bytes beyond max_bytes are aborted mid-stream with a size-limit error."""

    def handler(req):
        # Stream more than the cap; the reader must stop early.
        big = b"\x00" * 200
        return httpx.Response(200, content=big)

    monkeypatch.setattr(httpx, "AsyncClient", _client_with(handler))
    with pytest.raises(FetchError, match="size limit"):
        asyncio.run(fetch_image("http://x.test/huge", max_bytes=64))


def test_fetch_image_empty_body(monkeypatch):
    """A 200 with an empty body is treated as an error (nothing to decode)."""
    monkeypatch.setattr(httpx, "AsyncClient", _client_with(lambda req: httpx.Response(200, content=b"")))
    with pytest.raises(FetchError, match="empty"):
        asyncio.run(fetch_image("http://x.test/empty"))


# ---------------------------------------------------------------------------
# _resolve_image — endpoint integration via /panels/candidates (pure geometry)
# ---------------------------------------------------------------------------


def _settings():
    """Fresh Settings so the test isn't affected by a stray .env key."""
    return Settings()


def test_url_field_works_end_to_end(monkeypatch):
    """Posting ``url`` downloads the image and feeds it into the endpoint."""
    img = _png_bytes()
    # Stub the downloader at its import site in app.main so the endpoint gets
    # our bytes without any network call.
    async def fake_fetch(url, *, timeout, max_bytes):
        assert url == "http://store/a.png"
        return img

    # _resolve_image imports fetch_image lazily from .fetch, so patch there.
    import app.fetch as fetch_mod
    monkeypatch.setattr(fetch_mod, "fetch_image", fake_fetch)
    monkeypatch.setattr(main_mod, "_settings", lambda: _settings())

    c = TestClient(main_mod.app)
    r = c.post("/panels/candidates", data={"url": "http://store/a.png"})
    assert r.status_code == 200
    body = r.json()
    assert body["width"] == 8 and body["height"] == 8  # the 8x8 test PNG


def test_file_takes_precedence_over_url(monkeypatch):
    """When both file and url are given, the file is used (backward compatible)."""

    async def fake_fetch(url, *, timeout, max_bytes):  # pragma: no cover — must NOT run
        raise AssertionError("fetch_image should not be called when file is provided")

    import app.fetch as fetch_mod
    monkeypatch.setattr(fetch_mod, "fetch_image", fake_fetch)
    monkeypatch.setattr(main_mod, "_settings", lambda: _settings())

    c = TestClient(main_mod.app)
    # Upload a 10x10 PNG via file AND pass a url; the file wins.
    buf = io.BytesIO()
    Image.new("RGB", (10, 10)).save(buf, format="PNG")
    r = c.post(
        "/panels/candidates",
        files={"file": ("x.png", buf.getvalue(), "image/png")},
        data={"url": "http://store/a.png"},
    )
    assert r.status_code == 200
    assert r.json()["width"] == 10  # came from the file, not the stub


def test_neither_file_nor_url_is_400(monkeypatch):
    """Omitting both inputs is a client error, not a 500."""
    monkeypatch.setattr(main_mod, "_settings", lambda: _settings())
    c = TestClient(main_mod.app)
    r = c.post("/panels/candidates")
    assert r.status_code == 400
    assert "file" in r.json()["detail"] or "url" in r.json()["detail"]
