"""Feishu (Lark) custom-bot webhook delivery — ops alerts for slow/failed runs.

When ``OCR_FEISHU_WEBHOOK_URL`` is set, the perf monitor (``app/perf_alert.py``)
pushes a text message to a Feishu group bot when ``/analyze`` runs unusually
long or fails outright. This module is the transport only — it knows nothing
about thresholds or cooldowns.

Design notes (mirrors ``app/webhook.py`` so the two outbound HTTP paths read
the same way):
- **Synchronous function**: ``send_text`` runs inside the webhook thread pool
  in main.py. The caller never awaits the outcome — delivery is
  fire-and-forget; failures are logged, not raised, because an alerting hiccup
  must never affect the OCR result (or the alerting of the next request).
- **Bot signing (加签)**: Feishu custom bots configured with a signing secret
  require ``timestamp`` + ``sign`` fields in the JSON body. The algorithm is
  unusual: the HMAC-SHA256 *key* is ``f"{timestamp}\\n{secret}"``, the message
  is empty, and the digest is base64 (not hex). No secret → no signing fields,
  so the field doubles as the opt-in for auth.
- **No SSRF guard**: the URL comes from our own ``.env``, not from callers
  (see ``app/fetch.py`` docstring for the same trade-off).
- **Single attempt**: v1 does not retry — a missed alert beats a retry storm
  wedging a worker thread. The cooldown in perf_alert.py already rate-limits
  how often we send.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import time

import httpx

logger = logging.getLogger(__name__)

# Default per-delivery timeout. Alerts go to open.feishu.cn over the public
# internet; 10s is generous without letting a dead endpoint pin a worker.
DEFAULT_TIMEOUT: float = 10.0


def gen_sign(timestamp: int, secret: str) -> str:
    """Feishu custom-bot signature for the given ``timestamp`` (epoch seconds).

    Per Feishu's docs the string ``f"{timestamp}\\n{secret}"`` is used as the
    HMAC-SHA256 key over an EMPTY message, and the digest is base64-encoded
    (the webhook.py HMAC over the raw body is a different scheme entirely).
    The ``timestamp`` sent in the body must be the same value passed here, as
    a decimal string — Feishu rejects messages whose sign is older than 1h.
    """
    string_to_sign = f"{timestamp}\n{secret}"
    hmac_code = hmac.new(
        string_to_sign.encode("utf-8"), digestmod=hashlib.sha256
    ).digest()
    return base64.b64encode(hmac_code).decode("utf-8")


def send_text(
    text: str,
    webhook_url: str,
    secret: str = "",
    timeout: float = DEFAULT_TIMEOUT,
) -> bool:
    """POST a ``msg_type=text`` message to the Feishu bot. Never raises.

    Returns True when Feishu accepted the message (HTTP 2xx), False otherwise
    — handy for tests; production callers are fire-and-forget and ignore it.
    """
    if not webhook_url:
        return False

    body: dict = {"msg_type": "text", "content": {"text": text}}
    if secret:
        # Body timestamp must be a STRING and match the one baked into the sign.
        ts = int(time.time())
        body["timestamp"] = str(ts)
        body["sign"] = gen_sign(ts, secret)

    payload = json.dumps(body).encode("utf-8")
    headers = {"Content-Type": "application/json"}

    t0 = time.perf_counter()
    try:
        with httpx.Client(timeout=timeout, follow_redirects=False) as client:
            resp = client.post(webhook_url, content=payload, headers=headers)
        if resp.status_code >= 400:
            logger.warning(
                "feishu alert returned HTTP %s in %.2fs",
                resp.status_code, time.perf_counter() - t0,
            )
            return False
        logger.info("feishu alert delivered in %.2fs", time.perf_counter() - t0)
        return True
    except Exception as exc:  # noqa: BLE001 — fire-and-forget: never propagate
        logger.warning(
            "feishu alert failed in %.2fs: %s", time.perf_counter() - t0, exc
        )
        return False
