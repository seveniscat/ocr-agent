# ocr-agent

High-resolution OCR + polygon bbox engine for **packaging dielines** (100–10000px).
Outputs structured JSON: every text line / QR / barcode → polygon bbox + content + type + confidence. Optional annotated image for visual QA.

## Design in one paragraph
**Hybrid**: deterministic work (bbox detection, OCR, QR decode) is done by fast local expert models (PaddleOCR 3.x / PP-OCRv6 polygon detector + pyzbar); the cloud VLM (Qwen-VL) plays two roles — (1) a **fallback** for art / curved text PaddleOCR can't read confidently, and (2) the **AI-Native understanding layer** (`POST /understand`) that looks at the whole image and answers "what is this" (category / style / salient elements). Very high-resolution images are handled by **dynamic grid tiling** with overlap and cross-tile NMS deduplication — no single model ingests 10000px natively. The "AINative 审查" layer (probability sum, six-finger, compliance) is intentionally left as a pluggable interface in v1.

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

  ── separate branch (AI Native, level 1) ──
  image → [downscale to VLM sweet spot] → VLM "what is this"
        → UnderstandingResult { category, style, key_elements, summary }
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

# AI-Native understanding: "what is this image?" (one whole-image VLM call)
curl -F "file=@sample.png" http://localhost:8000/understand | jq
```

### Response shape (OCR)

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

### Response shape (understanding)

```jsonc
{
  "category": "食品-饮料",
  "category_confidence": 0.9,
  "panel_count_estimate": 6,
  "style_keywords": ["极简", "高饱和"],
  "dominant_colors": ["#E63946", "#F1FAEE"],
  "key_elements": [
    { "kind": "logo", "description": "品牌 logo", "location": [0.1, 0.1, 0.2, 0.1] },
    { "kind": "nutrition_table", "description": "营养成分表" }
  ],
  "summary": "一款饮料的纸盒展开图",
  "raw_note": null,                            // set when JSON parse degraded
  "model": "qwen"
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
| `OCR_UNDERSTAND_ENABLED` | `true` | Toggle the AI understanding layer (`/understand`) |
| `OCR_UNDERSTAND_MAX_SIDE` | `1080` | Long edge (px) image is downscaled to before the VLM |
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
  main.py          FastAPI routes (POST /analyze, /understand, /panels, /preprocess, GET /tasks/{id}, /healthz)
  pipeline.py      Orchestrator: tiles → OCR → codes → VLM → dedupe → annotate
  understanding.py AI-Native understanding layer: VLM "what is this" (level 1)
  tiling.py        Grid planning, coordinate remap, IoU NMS, dedupe
  panels.py        Die-line → main box-face panels (LSD long-line detection)
  preprocess.py    Die-line blank-margin auto-crop
  config.py        pydantic-settings (.env-driven)
  schemas.py       API I/O models
  ocr/             PaddleOCR wrapper (detector + recognizer + aggregator)
  codes/           pyzbar QR/barcode
  vlm/             Provider abstraction + Qwen-VL (recognize_crop + ask_image)
  viz/             Polygon annotator (debug/QA overlay)
  checks/          AINative reviewer plugin interface (v1: empty)
tests/             tiling + smoke + preprocess + understanding
samples/           Drop dieline images here for manual testing
```

## Known limits / v2 roadmap

- **PaddleOCR VRAM on 10000px tiles**: tile size is tunable (`OCR_TILE_TARGET_SIZE`); reduce if you hit OOM, or cap concurrency.
- **Extreme art text** may evade polygon detection → v2 adds a "suspicious empty region" second pass that asks the VLM to scan crops.
- **Curved / artistic QR codes** that pyzbar misses → v2 adds a WeChat-QRCode fallback (interface preserved in `codes/qrcode.py`).
- **PDF input**: v1 accepts bitmap images only. Add `pdf2image` rasterization at the entry point if needed (one-liner).

## AI-Native roadmap — from OCR engine to "image understanding master"

The understanding layer (`POST /understand`, level 1) is the first step of a
graduated path. Each level is additive — none rewrites the previous.

- **Level 1 ✅ shipped** — whole-image understanding: VLM downscales the image
  and answers "what is this" (category / style / key elements). Validates that
  the VLM can stand at the understanding layer.
- **Level 2** — per-panel semantics: reuse `panels.split_panels()` to cut 5–6
  panels, run one VLM call per panel, merge into panel-level understanding
  (which face is the front / nutrition / ingredients). `level=2` query param.
- **Level 3** — key-info extraction: per-panel VLM calls switch to structured
  prompts (product name / net weight / manufacturer present?), schema extends.
- **Level 4** — QA agent: `UnderstandingResult` + OCR `items[]` feed an LLM
  agent that runs a category-driven checklist through `checks/` and emits
  `CheckResult[]` (severity / location / evidence / suggestion). The
  `checks/base.py` hook finally lands.
- **Level 5** — fully tool-ified: understanding + OCR + codes + panels register
  as tools; the VLM/LLM orchestrates them autonomously.

### Large-image note for the VLM
VLMs effectively resolve ~2Kpx regardless of nominal resolution. The project's
existing tiling / panels / preprocess infrastructure is *also* the answer for
feeding large images to the VLM: level 1 downscales once (enough to "see what
it is"); level 2+ uses `panels.split_panels()` to feed each ~1500px panel
separately. No new large-image machinery is needed.

## Not in scope (v1)
- Web visualization UI (only JSON + annotated image).   ← partially delivered: the embedded UI now renders understanding cards
- Single-character-level polygons (granularity is line / semantic block).
- Any checker implementation (interface only).
