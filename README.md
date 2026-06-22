# ocr-agent

High-resolution OCR + polygon bbox engine for **packaging dielines** (100–10000px).
Outputs structured JSON: every text line / QR / barcode → polygon bbox + content + type + confidence. Optional annotated image for visual QA.

## Design in one paragraph
**Hybrid**: deterministic work (bbox detection, OCR, QR decode) is done by fast local expert models (PaddleOCR 3.x / PP-OCRv6 polygon detector + pyzbar); the cloud VLM (Qwen-VL) is used **only** as a fallback for art / curved text PaddleOCR can't read confidently. Very high-resolution images are handled by **dynamic grid tiling** with overlap and cross-tile NMS deduplication — no single model ingests 10000px natively. The "AINative 审查" layer (probability sum, six-finger, compliance) is intentionally left as a pluggable interface in v1.

## Architecture

```
image (100–10000px)
  │
  ├─ plan_grid → tiles (overlap 15%) ──────────────┐
  │                                                │
  └─ per tile:                                     │
       PaddleOCR det+rec  → text  (polygon, tile-local)
       pyzbar            → qr/barcode (polygon, tile-local)
  │
  ├─ offset polygon → global coords
  ├─ VLM fallback for low-confidence text crops
  ├─ dedupe (polygon IoU NMS + text similarity)
  └─ (optional) annotate → return JSON + image
```

## Quickstart

```bash
# 1. System libs
brew install zbar                     # macOS
# apt-get install libzbar0             # Debian/Ubuntu

# 2. Python deps
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
#   → at minimum set OCR_VLM_API_KEY if you want art-text fallback,
#     or set OCR_VLM_ENABLED=false to skip it.

# 4. Run
uvicorn app.main:app --reload --port 8000
```

### Use it

```bash
# Small/medium image → synchronous
curl -F "file=@sample.png" http://localhost:8000/analyze > result.json

# With annotated overlay
curl -F "file=@sample.png" "http://localhost:8000/analyze?annotate=1" | jq -r .annotated_image_b64 | base64 -d > annotated.png

# Large image (> OCR_LARGE_IMAGE_THRESHOLD) → 202 + task_id
curl -F "file=@huge.png" http://localhost:8000/analyze   # → { "task_id": "..." }
curl http://localhost:8000/tasks/<task_id>
```

### Response shape

```jsonc
{
  "image_meta": { "width": 8000, "height": 6000, "tile_count": 20 },
  "items": [
    {
      "id": "t_001",
      "type": "text",                          // text | art_text | qr | barcode
      "text": "净含量：250ml",
      "polygon": [[x1,y1],[x2,y2],[x3,y3],[x4,y4]],   // original-image coords
      "bbox":     [x1, y1, x2, y2],                   // AABB, derived
      "confidence": 0.97,
      "source": "paddleocr"                    // paddleocr | vlm_fallback | pyzbar
    }
  ],
  "annotated_image_b64": null                  // set when ?annotate=1
}
```

## Configuration (`.env`)

| Var | Default | Meaning |
|---|---|---|
| `OCR_TILE_TARGET_SIZE` | `1600` | Tile long-edge target px |
| `OCR_TILE_OVERLAP` | `0.15` | Overlap ratio between tiles |
| `OCR_SMALL_IMAGE_THRESHOLD` | `2000` | Below this on both dims → no tiling |
| `OCR_OCR_THRESHOLD` | `0.3` | PaddleOCR detection threshold |
| `OCR_REC_CONFIDENCE_FALLBACK` | `0.6` | Below this → route crop to VLM |
| `OCR_VLM_ENABLED` | `true` | Toggle art-text fallback |
| `OCR_VLM_PROVIDER` | `qwen` | VLM provider (qwen) |
| `OCR_VLM_API_KEY` | _empty_ | Required if VLM enabled |
| `OCR_VLM_BASE_URL` | DashScope | OpenAI-compatible endpoint |
| `OCR_VLM_MODEL` | `qwen-vl-max` | Model name |
| `OCR_LARGE_IMAGE_THRESHOLD` | `4000` | Long edge above → async (202) |

## Tests

```bash
pytest -q          # tiling geometry + API smoke; no paddle/pyzbar needed
```

`tests/test_tiling.py` covers the highest-bug-density logic:
- Grid coverage (no pixel gaps), edge flush, neighbour overlap.
- Tile-local → global coordinate remap (round-trip with a synthetic marker).
- Polygon IoU + deduplication merge rules (same-text merge, different-text keep).

## Project layout

```
app/
  main.py          FastAPI routes (POST /analyze, GET /tasks/{id}, /healthz)
  pipeline.py      Orchestrator: tiles → OCR → codes → VLM → dedupe → annotate
  tiling.py        Grid planning, coordinate remap, IoU NMS, dedupe
  config.py        pydantic-settings (.env-driven)
  schemas.py       API I/O models
  ocr/             PaddleOCR wrapper (detector + recognizer + aggregator)
  codes/           pyzbar QR/barcode
  vlm/             Provider abstraction + Qwen-VL impl
  viz/             Polygon annotator (debug/QA overlay)
  checks/          AINative reviewer plugin interface (v1: empty)
tests/             tiling + smoke
samples/           Drop dieline images here for manual testing
```

## Known limits / v2 roadmap

- **PaddleOCR VRAM on 10000px tiles**: tile size is tunable (`OCR_TILE_TARGET_SIZE`); reduce if you hit OOM, or cap concurrency.
- **Extreme art text** may evade polygon detection → v2 adds a "suspicious empty region" second pass that asks the VLM to scan crops.
- **Curved / artistic QR codes** that pyzbar misses → v2 adds a WeChat-QRCode fallback (interface preserved in `codes/qrcode.py`).
- **PDF input**: v1 accepts bitmap images only. Add `pdf2image` rasterization at the entry point if needed (one-liner).
- **AINative reviewers** ("probability sum ≠ 1", "six-finger", compliance) → plug into `app/checks/`. The hook (`Checker` / `run_all`) is already in place.

## Not in scope (v1)
- Web visualization UI (only JSON + annotated image).
- Single-character-level polygons (granularity is line / semantic block).
- Any checker implementation (interface only).
