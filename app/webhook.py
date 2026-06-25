"""Outbound webhook delivery — notify a business system when a detection finishes.

When an ``/analyze`` request carries a ``callback_url``, the pipeline fires a
small status-only payload to that URL on completion (or failure). The business
system then GETs ``/tasks/{task_id}`` to pull the full result — this keeps the
callback body tiny and idempotent regardless of how large the OCR output is.

Design notes (mirrors ``app/fetch.py`` so the two HTTP paths read the same way):
- **Synchronous function**: ``deliver`` runs inside a dedicated thread pool
  (``_webhook_executor`` in main.py), separate from the OCR pool, so a slow
  / hung callback endpoint can never starve an OCR worker. The caller never
  awaits the outcome — delivery is fire-and-forget; failures are logged, not
  raised, because a webhook hiccup must never affect the OCR result.
- **HMAC-SHA256 signing**: when the caller supplies ``callback_secret``, the
  exact JSON bytes we POST are signed and the hex digest is sent in
  ``X-Webhook-Signature: sha256=<hex>``. The receiver recomputes over the raw
  request body and compares (constant-time via ``compare_digest``). No secret →
  no signature header, so the field doubles as the opt-in for auth.
- **No SSRF guard**: callers are trusted internal systems (see ``app/fetch.py``
  docstring). If this service is ever exposed publicly, add an allow-host check.
- **Single attempt**: v1 does not retry. The receiver should be idempotent and
  fall back to polling ``/tasks/{id}`` if a callback is missed; a retry queue
  is noted as future work in the README.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from datetime import datetime, timezone
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# Default per-delivery timeout. Callbacks are status pings to internal systems;
# 10s is generous without letting a dead endpoint pin a worker thread forever.
DEFAULT_TIMEOUT: float = 10.0


def build_payload(
    task_id: str,
    status: str,
    biz_id: Optional[str] = None,
    error: Optional[str] = None,
    *,
    now: Optional[float] = None,
) -> dict:
    """Construct the outbound webhook body.

    Args:
        task_id: The OCR task id the receiver uses to GET /tasks/{task_id}.
        status: One of ``"done"`` / ``"error"`` — matches ``TaskStatus.status``.
        biz_id: Optional caller-supplied business id, echoed verbatim so the
            receiver can correlate the callback to its own order/record.
        error: Short error string; only set on failure.
        now: Override for ``time.time()`` (tests only).

    The body is deliberately tiny and always has the same keys (``error`` /
    ``biz_id`` omitted only when empty), so receivers can deserialize into a
    fixed schema without conditional parsing.
    """
    event = "analyze.completed" if status == "done" else "analyze.failed"
    ts = datetime.fromtimestamp(now if now is not None else time.time(), tz=timezone.utc)
    payload: dict = {
        "event": event,
        "task_id": task_id,
        "status": status,
        "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    if biz_id:
        payload["biz_id"] = biz_id
    if error:
        payload["error"] = error
    return payload


def sign(body: bytes, secret: str) -> str:
    """HMAC-SHA256 of the raw request body, formatted for the signature header.

    Returns ``"sha256=<hex>"`` so the algorithm is self-describing in the header
    (room to add more algorithms later without a header-name change). The
    receiver must sign the *exact* bytes it received — i.e. the raw body, not a
    re-serialized JSON object (key order / whitespace would diverge).
    """
    digest = hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def deliver(
    task_id: str,
    status: str,
    biz_id: Optional[str],
    callback_url: str,
    secret: Optional[str],
    error: Optional[str] = None,
    *,
    timeout: float = DEFAULT_TIMEOUT,
) -> None:
    """POST the status payload to ``callback_url``. Never raises.

    Runs on the webhook thread pool. Any failure (network error, non-2xx,
    timeout) is swallowed and logged at WARNING — the OCR task that triggered
    it has already succeeded (or already recorded its own error), so a callback
    delivery problem must not propagate.
    """
    body = json.dumps(build_payload(task_id, status, biz_id, error)).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if secret:
        headers["X-Webhook-Signature"] = sign(body, secret)

    t0 = time.perf_counter()
    try:
        with httpx.Client(timeout=timeout, follow_redirects=False) as client:
            resp = client.post(callback_url, content=body, headers=headers)
        if resp.status_code >= 400:
            logger.warning(
                "webhook task=%s to %s returned HTTP %s in %.2fs",
                task_id, callback_url, resp.status_code, time.perf_counter() - t0,
            )
        else:
            logger.info(
                "webhook task=%s delivered (%s) to %s in %.2fs",
                task_id, status, callback_url, time.perf_counter() - t0,
            )
    except Exception as exc:  # noqa: BLE001 — fire-and-forget: never propagate
        logger.warning(
            "webhook task=%s to %s failed in %.2fs: %s",
            task_id, callback_url, time.perf_counter() - t0, exc,
        )
