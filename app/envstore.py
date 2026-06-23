"""Thin wrapper around python-dotenv for persisting OCR_* settings to .env.

Persists VLM configuration edited through the Web UI. Only writes keys we own,
one key at a time via :func:`dotenv.set_key` — never rewrites the whole file, so
comments and unknown keys are preserved. The file is created if missing.

``mask()`` produces a non-reversible display form for API keys so the GET
endpoint can tell the UI "a key is set" without ever leaking the secret.
"""
from __future__ import annotations

from .config import ENV_PATH  # single source of truth for the .env location


def upsert(key: str, value: str) -> None:
    """Write ``key=value`` into .env, creating the file if it doesn't exist.

    Uses :func:`dotenv.set_key` which updates a single key in place (preserving
    everything else in the file). ``value`` is stored verbatim; callers are
    responsible for not passing secrets they don't want stored.
    """
    from dotenv import set_key

    set_key(str(ENV_PATH), key, value)


def mask(secret: str) -> str:
    """Non-reversible display form for a secret, for safe echoing to the UI.

    Empty → "". Very short (≤ 8 chars) → fully masked (no partial leak). Else
    "first2***last4" (e.g. ``sk-abcd1234`` → ``sk***1234``).
    """
    if not secret:
        return ""
    if len(secret) <= 8:
        return "***"
    return f"{secret[:2]}***{secret[-4:]}"
