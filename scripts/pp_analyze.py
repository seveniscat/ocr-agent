#!/usr/bin/env python3
"""Pretty-print an OCR /analyze JSON response (piped from curl via Makefile).

Reads the API response from stdin and prints a compact human-readable summary:
total count, per-type counts, and each item's type + text/content + bbox.

Usage:
    curl -s ... | python scripts/pp_analyze.py
    curl -s ... | python scripts/pp_analyze.py --raw   # full JSON, indented
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter


def main() -> int:
    parser = argparse.ArgumentParser(description="Pretty-print an /analyze response.")
    parser.add_argument(
        "--raw", action="store_true",
        help="print the full JSON indented instead of the summary",
    )
    args = parser.parse_args()

    raw = sys.stdin.read()
    if not raw.strip():
        print("✗ 空响应(服务可能没返回数据,检查服务是否在跑)", file=sys.stderr)
        return 1

    try:
        d = json.loads(raw)
    except json.JSONDecodeError as exc:
        print(f"✗ 不是合法 JSON: {exc}", file=sys.stderr)
        print(f"  原始响应前 300 字符:{raw[:300]!r}", file=sys.stderr)
        return 1

    # Surface API errors (e.g. {"detail": "provide an image via 'file' or 'url'"})
    if "detail" in d and "items" not in d:
        print(f"✗ API 错误:{d['detail']}", file=sys.stderr)
        return 1

    if args.raw:
        print(json.dumps(d, indent=2, ensure_ascii=False))
        return 0

    items = d.get("items", [])
    by_type = Counter(i["type"] for i in items)
    meta = d.get("image_meta", {})

    print(f"图片 {meta.get('width')}×{meta.get('height')}  |  识别 {len(items)} 项")
    if by_type:
        print("类型分布:" + "  ".join(f"{k}={v}" for k, v in by_type.items()))
    print("-" * 60)
    for i in items:
        text = i.get("text") or i.get("content") or ""
        box = i.get("bbox")
        box_s = f"[{int(box[0])},{int(box[1])},{int(box[2])},{int(box[3])}]" if box else ""
        print(f"  [{i['type']:9}] {text:30} {box_s}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
