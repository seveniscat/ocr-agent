"""Tests for the QR/barcode code engine.

Focus on the two error-prone parts of the upscale-retry logic:
1. Coordinate scaling — an upscaled re-detection maps back to native space.
2. De-duplication — a code that decoded at both native and upscaled scale is
   kept once, while a genuinely different code at the same spot is kept too.

These run without generating a real QR image (the `qrcode` lib isn't a dep);
they exercise CodeDetection objects + the dedup helper directly. The end-to-end
"small QR decodes only after upscale" behavior is verified manually against a
real screenshot (see commit message), not as a unit test.
"""
from __future__ import annotations

from app.codes.qrcode import CodeDetection, _is_duplicate_code


def _det(content: str, x1, y1, x2, y2, type_="qr") -> CodeDetection:
    return CodeDetection(
        type=type_,
        content=content,
        polygon=[[x1, y1], [x2, y1], [x2, y2], [x1, y2]],
    )


def test_dedup_same_content_overlapping_is_duplicate():
    """Same payload + overlapping polygon → duplicate (the upscaled re-hit)."""
    native = _det("https://x.com", 100, 100, 160, 160)
    upscaled = _det("https://x.com", 98, 99, 162, 161)  # ~same spot
    assert _is_duplicate_code(upscaled, [native]) is True


def test_dedup_different_content_same_spot_not_duplicate():
    """Different payload at the same spot → keep both (never silently drop)."""
    a = _det("https://x.com", 100, 100, 160, 160)
    b = _det("https://y.com", 100, 100, 160, 160)
    assert _is_duplicate_code(b, [a]) is False


def test_dedup_same_content_disjoint_not_duplicate():
    """Same payload but at a totally different location → two real codes."""
    a = _det("https://x.com", 100, 100, 160, 160)
    b = _det("https://x.com", 500, 500, 560, 560)
    assert _is_duplicate_code(b, [a]) is False


def test_dedup_empty_existing_not_duplicate():
    assert _is_duplicate_code(_det("x", 0, 0, 10, 10), []) is False


def test_dedup_multiple_existing_finds_match():
    """Match against any existing entry, not just the first."""
    existing = [
        _det("a", 0, 0, 10, 10),
        _det("b", 0, 0, 10, 10),
        _det("c", 200, 200, 260, 260),
    ]
    cand = _det("c", 202, 201, 258, 259)
    assert _is_duplicate_code(cand, existing) is True
