"""Fetch an image from a URL into bytes (for the ``url`` form field).

The OCR endpoints accept either a multipart ``file`` upload or a ``url`` form
field. This module turns the URL into ``bytes`` so the rest of the pipeline —
which already consumes ``bytes`` (see ``tiling.load_image`` /
``Pipeline.run``) — needs no changes.

Design notes:
- **Streaming + size cap**: we read the body in chunks and abort as soon as the
  running total exceeds ``max_bytes``, so a multi-GB URL can't exhaust memory.
- **Non-blocking**: ``httpx.AsyncClient`` keeps the event loop free during the
  (possibly slow) download of large die-line images.
- **No FastAPI dependency**: a plain ``async`` function, easy to unit-test with
  ``httpx.MockTransport`` (no extra test deps).
- **No SSRF guard**: callers are trusted internal systems (see README). If this
  service is ever exposed externally, add an allow-host check here.
"""
from __future__ import annotations

import httpx


class FetchError(Exception):
    """Raised when a URL fetch fails (non-2xx, timeout, or size limit).

    Carries a short, caller-facing message; the endpoint maps it to HTTP 400.
    """


# Chunk size for the streaming read. Small enough to bound memory and check the
# size cap frequently; large enough to avoid per-call overhead on big images.
_CHUNK = 64 * 1024


async def fetch_image(
    url: str,
    *,
    timeout: float = 30.0,
    max_bytes: int = 104_857_600,
) -> bytes:
    """Download ``url`` and return its bytes.

    Args:
        url: HTTP(S) URL of the image.
        timeout: Connect/read timeout in seconds.
        max_bytes: Abort with :class:`FetchError` once the body exceeds this.

    Raises:
        FetchError: on non-2xx status, timeout/network error, or size overrun.
    """
    # follow_redirects: internal image stores sometimes 302 to a CDN path.
    try:
        async with httpx.AsyncClient(
            timeout=timeout, follow_redirects=True
        ) as client:
            async with client.stream("GET", url) as resp:
                if resp.status_code >= 400:
                    raise FetchError(f"URL returned HTTP {resp.status_code}")
                buf = bytearray()
                async for chunk in resp.aiter_bytes(_CHUNK):
                    buf.extend(chunk)
                    if len(buf) > max_bytes:
                        raise FetchError(
                            f"image exceeds size limit ({max_bytes} bytes)"
                        )
                if not buf:
                    raise FetchError("URL returned an empty body")
                return bytes(buf)
    except FetchError:
        raise  # already a caller-facing error — re-raise unchanged
    except httpx.TimeoutException as exc:
        raise FetchError(f"download timed out after {timeout}s") from exc
    except httpx.HTTPError as exc:
        # Network errors (connection refused, DNS, etc.) — keep it short.
        raise FetchError(f"download failed: {exc}") from exc
