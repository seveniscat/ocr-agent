#!/usr/bin/env python3
"""Diagnose where text boxes get LOST, layer by layer.

For a given image, prints the box count and contents at each stage of the
pipeline so you can see exactly which layer drops the missing text:

  L1  detector dt_polys      — what the text detector boxed (script-agnostic)
  L2  recognizer rec_polys   — what survived the rec score filter
  L3  OCREngine output       — our detect_and_recognize (incl. recovered crops)
  L4  merge_same_line_overlaps — after same-line overlap merge
  L5  dedupe_items           — after cross-tile/seam dedupe

If a missing text is absent at L1 → the DETECTOR never saw it (detection
problem: try lower ocr_threshold, higher unclip, or preprocessing).
If it's at L1 but gone at L2 → the RECOGNIZER filtered it (low score).
If it's at L2 but gone at L3+ → our POST-PROCESSING dropped it (a bug here).

Usage:
    PYTHONPATH=. .venv/bin/python scripts/debug_layers.py samples/xxx.jpeg
    PYTHONPATH=. .venv/bin/python scripts/debug_layers.py path/to/your.png
    PYTHONPATH=. .venv/bin/python scripts/debug_layers.py URL            # http(s)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


def _load(path_or_url: str) -> np.ndarray:
    from app.tiling import load_image

    if path_or_url.startswith(("http://", "https://")):
        import io
        import urllib.request

        with urllib.request.urlopen(path_or_url, timeout=30) as r:  # noqa: S310
            return load_image(r.read())
    return load_image(Path(path_or_url).read_bytes())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image", help="image path or http(s) URL")
    ap.add_argument(
        "--granularity", default="line", choices=["word", "line", "paragraph"],
        help="output granularity (default: line)",
    )
    args = ap.parse_args()

    from app.config import Settings
    from app.ocr.detector import OCREngine
    from app.schemas import Item
    from app.tiling import (
        dedupe_items, merge_same_line_overlaps, polygon_to_bbox,
    )

    s = Settings()
    img = _load(args.image)
    h, w = img.shape[:2]
    print(f"image: {w}x{h}  granularity={args.granularity}")
    print("=" * 72)

    eng = OCREngine(s)
    eng._ensure_loaded()

    # ---- L1/L2: raw PaddleOCR layers ----
    raw = eng._ocr.predict(img)
    r0 = raw[0]
    dt = [p for p in (r0.get("dt_polys") or []) if p is not None and len(p) >= 4]
    rec_p = [p for p in (r0.get("rec_polys") or []) if p is not None and len(p) >= 4]
    rec_t = r0.get("rec_texts") or []
    rec_s = r0.get("rec_scores") or []
    print(f"L1 detector dt_polys:    {len(dt)} boxes")
    print(f"L2 recognizer rec_polys: {len(rec_p)} boxes  "
          f"(dropped by rec filter: {len(dt) - len(rec_p)})")
    print("-" * 72)
    print("L2 recognized contents (score | text):")
    for t, sc in zip(rec_t, rec_s):
        mark = "  " if sc and sc > 0 else "!!"
        print(f"  {mark}[{float(sc):.3f}] {t!r}")
    print("-" * 72)
    print("L1 detector boxes (bbox only — what was seen, regardless of script):")
    for p in dt:
        xs = [pt[0] for pt in p[:4]]
        ys = [pt[1] for pt in p[:4]]
        print(f"  bbox=[{min(xs):.0f},{min(ys):.0f},{max(xs):.0f},{max(ys):.0f}]")

    # ---- L3: our engine (recovers dropped boxes as recognized=False) ----
    dets = eng.detect_and_recognize(img, granularity=args.granularity)
    rec = [d for d in dets if d.recognized]
    unrec = [d for d in dets if not d.recognized]
    print("=" * 72)
    print(f"L3 OCREngine output: recognized={len(rec)}  "
          f"recovered(unrecognized)={len(unrec)}")

    # ---- L4/L5: post-processing on Items (mirrors pipeline.run) ----
    items = [
        Item(
            id=f"tmp{i}", type="text", text=(d.text or None),
            polygon=d.polygon, bbox=polygon_to_bbox(d.polygon),
            confidence=d.confidence, source="paddleocr",
            granularity=d.granularity, recognized=d.recognized,
            crop_b64=d.crop_b64,
        )
        for i, d in enumerate(dets)
    ]
    merged = merge_same_line_overlaps(
        items,
        same_line_y_thres=s.tile_merge_y_thres,
        x_overlap_ratio=s.same_line_merge_x_overlap,
    )
    print(f"L4 merge_same_line_overlaps: {len(items)} -> {len(merged)}")
    deduped = dedupe_items(
        merged, merge_x_thres=s.tile_merge_x_thres, merge_y_thres=s.tile_merge_y_thres
    )
    print(f"L5 dedupe_items: {len(merged)} -> {len(deduped)}")
    print("=" * 72)
    print("FINAL items (after all layers):")
    for it in deduped:
        tag = "RECOGNIZED" if it.recognized else "UNRECOGNIZED"
        crop = f" crop={'yes' if it.crop_b64 else 'no'}" if not it.recognized else ""
        print(f"  [{tag}] [{it.confidence:.3f}] {it.text!r}{crop}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
