#!/usr/bin/env python3
"""Probe whether the installed paddlepaddle can run PaddleOCR PP-OCRv6 on CPU.

Standalone — no app imports. Generates a synthetic image with text and runs
PaddleOCR.predict() on it, catching the known 3.3.x PIR/oneDNN regression:

    NotImplementedError / ConvertPirAttribute2RuntimeAttribute not support

Exit codes:
    0  -> paddle runs PP-OCRv6 fine (safe to upgrade)
    1  -> paddle hits the PIR/oneDNN regression (keep <3.3)
    2  -> some other error (investigate)

Usage:
    .venv/Scripts/python.exe scripts/probe_paddle.py
"""
from __future__ import annotations

import sys
import traceback


def _make_image():
    """Synthesize a 640x200 white image with black text 'HELLO 2026 OCR'."""
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (640, 200), color="white")
    draw = ImageDraw.Draw(img)
    # Use default bitmap font (always available); size doesn't matter for probing.
    font = ImageFont.load_default()
    draw.text((40, 80), "HELLO 2026 OCR 12345", fill="black", font=font)
    return img


def main() -> int:
    # 1. Report versions up front.
    try:
        import paddle
        import paddleocr
        import paddlex
    except ImportError as exc:
        print(f"✗ paddle/paddleocr/paddlex not importable: {exc}", file=sys.stderr)
        return 2

    print(f"paddle      = {paddle.__version__}")
    print(f"paddleocr   = {paddleocr.__version__}")
    print(f"paddlex     = {paddlex.__version__}")
    print(f"cuda        = {getattr(paddle.version, 'cuda', lambda: 'n/a')()}")
    print()

    # 2. Build the engine with the same shape the app uses.
    try:
        from paddleocr import PaddleOCR
    except Exception:
        traceback.print_exc()
        return 2

    print("Loading PaddleOCR (PP-OCRv6, lang=ch)…", flush=True)
    try:
        ocr = PaddleOCR(
            lang="ch",
            ocr_version="PP-OCRv6",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=True,
        )
    except Exception:
        traceback.print_exc()
        return 2
    print("Engine loaded.", flush=True)

    # 3. Run predict() — this is where the 3.3.x bug bites.
    # PaddleOCR.predict() only accepts numpy.ndarray or str(path); PIL is ignored.
    import numpy as np
    img = np.array(_make_image())
    print("Running predict() on synthetic image (numpy uint8)…", flush=True)
    try:
        results = ocr.predict(img)
    except NotImplementedError as exc:
        msg = f"{exc}".lower()
        if "pir" in msg or "onednn" in msg or "runtimeattribute" in msg or msg:
            print()
            print("✗ NotImplementedError caught — this is the known 3.3.x")
            print("  PIR/oneDNN regression (issues #77340 / #18162).")
            print(f"  Exception: {exc!r}")
        return 1
    except Exception as exc:
        msg = f"{exc}".lower()
        if "pir" in msg or "onednn" in msg or "runtimeattribute" in msg:
            print()
            print("✗ PIR/oneDNN-related error caught — same regression,")
            print("  just a different exception type.")
            print(f"  {type(exc).__name__}: {exc}")
            return 1
        print()
        print(f"✗ Other error during predict(): {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return 2

    # 4. Inspect the result dict shape the app relies on.
    r0 = results[0] if results else {}
    polys = r0.get("rec_polys") or []
    texts = r0.get("rec_texts") or []
    scores = r0.get("rec_scores") or []
    print()
    print(f"✓ predict() returned. boxes={len(polys)} texts={len(texts)} scores={len(scores)}")
    for t, s in zip(texts, scores):
        try:
            print(f"    {float(s):.2f}  {t}")
        except (TypeError, ValueError):
            print(f"    {s!r}  {t}")
    print()
    print("PASS — paddle can run PP-OCRv6 on this machine.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
