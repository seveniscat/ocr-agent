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
#   → 或用封装好的 make 命令(见下文"运维命令"):`make up`
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

> 📄 **第三方接入**:完整接口说明见 [`docs/API.md`](docs/API.md)(端点、参数、响应结构、Python 示例、错误码、坐标约定)。

### Image input: file upload OR URL

Every image endpoint accepts either a multipart `file` **or** a `url` form
field (file wins when both are given — backward compatible). Use `url` to
avoid downloading + re-uploading when the image already lives on an internal
image store / CDN.

```bash
# Pass an image URL instead of uploading
curl -F "url=http://内网图床/sample.png" http://localhost:8000/analyze
curl -F "url=http://内网图床/sample.png" "http://localhost:8000/analyze?annotate=1"
```

### Webhook 回调(`callback_url`)

检测完成后,业务系统可以用 webhook 被动接收完成通知,**免去轮询 `/tasks/{id}`**。在 `/analyze`
请求里加 3 个可选表单字段即可(全部向后兼容,不传时行为不变):

| 字段 | 必填 | 说明 |
|---|---|---|
| `callback_url` | 否 | 回调地址,必须是 http(s)。检测完成(或失败)后服务会向这里 POST 一个轻量状态包。 |
| `callback_secret` | 否 | 共享密钥。传入则对 body 做 HMAC-SHA256 签名,放在请求头 `X-Webhook-Signature: sha256=<hex>`;不传则不签名。 |
| `biz_id` | 否 | 业务标识,原样回传到回调体,便于业务系统关联自己的订单/记录。 |

> 触发范围:`/analyze` 的**同步路径**(小图)和**异步路径**(大图)都会触发。同步路径在带回调时会额外生成 `task_id` 并把结果存入任务表,这样无论走哪条路径,业务系统都能统一用 `GET /tasks/{task_id}` 回查完整结果。

**回调体**(故意做得小且幂等 —— 完整 OCR 结果请用 `task_id` 回查):

```jsonc
{
  "event": "analyze.completed",        // 或 "analyze.failed"
  "task_id": "91074aa2c74d46a99ca035e45f8bbb59",
  "status": "done",                    // 或 "error"
  "biz_id": "order-42",                // 调用方传入才有
  "timestamp": "2026-06-24T12:34:56Z"  // ISO-8601 UTC
  // "error": "..."                    // 仅 failed 时出现
}
```

**示例 — 异步大图 + 回调:**

```bash
curl -F "file=@huge.png" \
     -F "callback_url=http://biz.test/hook" \
     -F "callback_secret=s3cr3t" \
     -F "biz_id=order-42" \
     http://localhost:8000/analyze
# → 202 { "task_id": "..." }  (立即返回)
# → 检测完成后服务主动 POST http://biz.test/hook
```

**接收方校验签名(Python 示例):**

```python
import hmac, hashlib
expected = "sha256=" + hmac.new(SECRET.encode(), request.raw_body, hashlib.sha256).hexdigest()
hmac.compare_digest(expected, request.headers["X-Webhook-Signature"])  # True = 合法
# 校验通过后再 GET /tasks/{task_id} 拉完整结果
```

> **投递策略**:fire-and-forget,单次尝试,10s 超时。投递在**独立线程池**里执行,**绝不影响 OCR 结果**(失败仅记 WARNING 日志)。回调失败的兜底方式是继续轮询 `/tasks/{id}` —— 接收方应做成幂等。重试队列/指数退避见 v2 roadmap。调用方为可信内网系统,故未加 SSRF 守卫(与 `url` 输入一致);若对外暴露请自行加白名单。


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

## 运维命令(`make`)

项目根目录的 `Makefile` 封装了常用运维操作,降低记忆负担。`make`(或 `make help`)列出全部命令。

```bash
make up          # 启动服务(对外监听 0.0.0.0:8000,后台常驻,关终端不掉)
make dev-up      # 开发模式(--reload 热重载,改 app/ 自动重启;前台运行)
make down        # 停止服务(--reload 的进程树也能彻底杀干净)
make restart     # 重启(down + up)
make status      # 查看进程 + 监听端口
make health      # 健康检查(本机 + 内网 IP)
make log         # 实时跟踪日志(Ctrl+C 退出)
make docs        # 浏览器打开 /docs 交互式文档
make pytest      # 跑测试套件
make clean       # 停服务 + 清日志/缓存
```

一键验证(启动后用图片 URL 跑一遍 OCR):

```bash
make test URL=https://你的图.png          # file 或 url 二选一
make test FILE=samples/xxx.jpeg
make annotate URL=https://你的图.png OUT=out.png   # 生成带标注框的图
make understand URL=https://你的图.png             # AI 理解"这张图是什么"
```

`make test` 会打印精简汇总(图片尺寸、按类型计数、每项的文字+bbox):

```
图片 849×831  |  识别 4 项
类型分布:text=4
------------------------------------------------------------
  [text     ] MARVEL                         [498,383,664,444]
  [text     ] VERSION                        [153,400,514,532]
```

常用变量(命令行覆盖):`HOST`(默认 `0.0.0.0`)、`PORT`(默认 `8000`)、`URL` / `FILE`(测试用图片)、`OUT`(标注图文件名)。

## Configuration (`.env`)

| Var | Default | Meaning |
|---|---|---|
| `OCR_TILE_TARGET_SIZE` | `1600` | Tile long-edge target px |
| `OCR_TILE_OVERLAP` | `0.15` | Overlap ratio between tiles |
| `OCR_SMALL_IMAGE_THRESHOLD` | `2000` | Below this on both dims → no tiling |
| `OCR_OCR_VERSION` | `PP-OCRv6` | OCR pipeline; v6 = `PP-OCRv6_medium_rec` (≈50-language single model) |
| `OCR_OCR_LANG` | `ch` | Lang tag; default `ch` + v6 = multilingual rec |
| `OCR_OCR_THRESHOLD` | `0.3` | PaddleOCR detection threshold |
| `OCR_REC_CONFIDENCE_FALLBACK` | `0.6` | Below this → VLM re-reads the crop text (bbox unchanged) |
| `OCR_VLM_ENABLED` | `true` | Toggle art-text fallback |
| `OCR_VLM_PROVIDER` | `qwen` | VLM provider (qwen) |
| `OCR_VLM_API_KEY` | _empty_ | Required if VLM enabled |
| `OCR_VLM_BASE_URL` | DashScope | OpenAI-compatible endpoint |
| `OCR_VLM_MODEL` | `qwen-vl-max` | Model name |
| `OCR_UNDERSTAND_ENABLED` | `true` | Toggle the AI understanding layer (`/understand`) |
| `OCR_UNDERSTAND_MAX_SIDE` | `1080` | Long edge (px) image is downscaled to before the VLM |
| `OCR_LARGE_IMAGE_THRESHOLD` | `4000` | Long edge above → async (202) |
| `OCR_URL_FETCH_TIMEOUT` | `30` | URL download connect/read timeout (seconds) |
| `OCR_URL_FETCH_MAX_BYTES` | `104857600` | Abort URL download once body exceeds this (100MB) |

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
  main.py          FastAPI routes (POST /analyze, /understand, /panels, /panels/vlm, /panels/candidates, /panels/compute, /agent/understand, GET /tasks/{id}, /healthz)
  pipeline.py      Orchestrator: tiles → OCR → codes → VLM → dedupe → annotate
  understanding.py AI-Native understanding layer: VLM "what is this" (level 1)
  tiling.py        Grid planning, coordinate remap, IoU NMS, dedupe
  panels.py        Die-line → main box-face panels (LSD long-line detection)
  preprocess.py    Die-line blank-margin auto-crop
  config.py        pydantic-settings (.env-driven)
  fetch.py         Download `url` form field → bytes (streamed + size-capped)
  webhook.py       Outbound webhook delivery (callback_url → HMAC-signed status ping)
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
