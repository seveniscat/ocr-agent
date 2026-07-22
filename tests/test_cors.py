"""CORS behavior tests.

Confirms:
- The configured allow-list Origin gets the right preflight + actual headers.
- The Access-Control-Max-Age header is sent so browsers cache preflights
  (cutting the failure window for "occasional CORS errors").
- An off-list Origin is NOT echoed back (no Allow-Origin header).

These tests don't depend on which Origins happen to be in OCR_CORS_ORIGINS —
CORSMiddleware always allows '*' if configured, and for an explicit list we
just need to use a known-listed origin. We read the configured list at test
time and pick the first entry; if CORS is disabled (empty) we skip.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def _client_and_origin():
    """Return (TestClient, a known-listed origin) or (None, None) if CORS is off."""
    from fastapi.testclient import TestClient

    from app.config import get_settings
    from app.main import app

    raw = (get_settings().cors_origins or "").strip()
    if not raw:
        return None, None
    origin = "*" if raw == "*" else [o.strip() for o in raw.split(",") if o.strip()][0]
    return TestClient(app), origin


def test_preflight_returns_cors_headers():
    """OPTIONS preflight from a listed Origin gets Access-Control-Allow-* back."""
    client, origin = _client_and_origin()
    if client is None:
        pytest.skip("CORS disabled (OCR_CORS_ORIGINS empty)")
    # The actual cross-origin request the browser would make to /analyze.
    resp = client.options(
        "/analyze",
        headers={
            "Origin": "http://example.com" if origin == "*" else origin,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    # CORSMiddleware answers preflight with 200 OK.
    assert resp.status_code == 200
    # The browser-required header: echoes the allowed Origin back.
    assert "access-control-allow-origin" in resp.headers
    # Methods + headers allowed for the actual follow-up request.
    assert "access-control-allow-methods" in resp.headers
    assert "access-control-allow-headers" in resp.headers


def test_preflight_returns_max_age():
    """Access-Control-Max-Age is set so browsers cache preflits (key change).

    Without this, the browser sends OPTIONS before every cross-origin POST.
    With max_age=600, it caches the preflight for 10 min, shrinking the window
    where a server restart or long OCR request can trip a 'CORS' error.
    """
    client, _ = _client_and_origin()
    if client is None:
        pytest.skip("CORS disabled (OCR_CORS_ORIGINS empty)")
    resp = client.options(
        "/analyze",
        headers={
            "Origin": "http://off-list-origin.example",
            "Access-Control-Request-Method": "POST",
        },
    )
    # max_age is sent regardless of whether the Origin is allowed (the header
    # is a hint about caching, not an authorization). We just assert presence
    # and a positive integer value.
    assert "access-control-max-age" in resp.headers
    assert int(resp.headers["access-control-max-age"]) > 0


def test_off_list_origin_not_echoed_on_actual_request():
    """A real (non-preflight) request from an off-list Origin gets no Allow-Origin.

    The browser's CORS check fails when Allow-Origin is absent — this is what
    blocks unwanted origins. We use /healthz (a cheap GET) to avoid running OCR.
    """
    client, origin = _client_and_origin()
    if client is None or origin == "*":
        pytest.skip("requires an explicit allow-list (not '*' and not disabled)")
    resp = client.get("/healthz", headers={"Origin": "http://definitely-not-allowed.example"})
    assert resp.status_code == 200  # the request itself succeeds
    # CORS spec: no Access-Control-Allow-Origin header = browser blocks it.
    assert "access-control-allow-origin" not in resp.headers


def test_listed_origin_echoed_on_actual_request():
    """A real request from a listed Origin gets Allow-Origin back."""
    client, origin = _client_and_origin()
    if client is None:
        pytest.skip("CORS disabled (OCR_CORS_ORIGINS empty)")
    # Use an Origin we know is allowed. For '*' the middleware echoes the
    # request Origin OR returns '*' (depends on allow_credentials), so we just
    # check the header is present.
    test_origin = "http://anything.example" if origin == "*" else origin
    resp = client.get("/healthz", headers={"Origin": test_origin})
    assert resp.status_code == 200
    assert "access-control-allow-origin" in resp.headers
