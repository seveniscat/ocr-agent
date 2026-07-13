# OCR 服务接口接入说明

面向第三方系统的 HTTP API 接入文档。所有接口均为 RESTful,支持图片上传(`file`)或图片 URL(`url`)两种输入方式。

---

## 1. 服务地址

| 环境 | 地址 |
|------|------|
| 内网访问 | `http://10.1.93.196:8000` |
| 交互式文档 | `http://10.1.93.196:8000/docs`(浏览器打开,可直接试调) |
| 健康检查 | `GET http://10.1.93.196:8000/healthz` |

---

## 2. 通用约定

### 2.1 图片输入方式(所有图片端点通用)

每个接收图片的端点都支持二选一的输入:

| 字段 | 位置 | 类型 | 说明 |
|------|------|------|------|
| `file` | multipart | 二进制 | 上传图片文件(png/jpg/jpeg/webp) |
| `url`  | form    | string | 图片 URL,服务端下载 |

**规则**:`file` 和 `url` 二选一。同时传时 **`file` 优先**;都不传返回 `400`。

```bash
# 方式一:上传文件
curl -F "file=@sample.png" http://10.1.93.196:8000/analyze

# 方式二:传 URL(内网图床已有图时更方便)
curl -F "url=http://内网图床/sample.png" http://10.1.93.196:8000/analyze
```

### 2.2 坐标系约定

| 字段 | 格式 | 坐标系 |
|------|------|--------|
| `bbox` | `[x1, y1, x2, y2]` | **原图像素**,左上角原点,X 向右、Y 向下 |
| `polygon` | `[[x,y],[x,y],[x,y],[x,y]]` | 4 点四边形,原图像素坐标 |

> ⚠️ **裁白边偏移**:默认会先裁掉刀模图四周白边,`bbox`/`polygon` 参考的是**裁白边后的图**。映回原图需加 `image_meta.crop` 偏移:
> ```python
> x0, y0 = result["image_meta"]["crop"][:2]
> orig_bbox = [bx1 + x0, by1 + y0, bx2 + x0, by2 + y0]
> ```
> 当 `crop` 为 `null`(无白边)时,坐标直接对原图,无需偏移。

### 2.3 同步与异步

| 图片长边 | 处理方式 | HTTP 响应 |
|----------|----------|-----------|
| ≤ 4000px | **同步** | `200` + 完整结果(直接在响应体) |
| > 4000px | **异步** | `202` + `task_id`(需轮询取结果) |

异步取结果:

```bash
# 第一步:发起,拿到 task_id
curl -F "url=http://..." http://10.1.93.196:8000/analyze
# → {"task_id":"f32332e8...","image_meta":{...},"items":[]}

# 第二步:轮询(间隔 5 秒),直到 status=done
curl http://10.1.93.196:8000/tasks/f32332e8...
# status: pending → running → done(result 字段即最终结果)
```

---

## 3. 接口清单

### 3.1 `POST /analyze` —— OCR 识别(核心接口)

识别图片中的**文字、艺术字、二维码、条码**,返回文字内容 + 坐标框。

#### 请求参数

| 参数 | 位置 | 必填 | 说明 |
|------|------|------|------|
| `file` / `url` | form | 二选一 | 图片输入(见 2.1) |
| `options` | form | 否 | OCR 参数覆盖,JSON 字符串(见下表) |
| `annotate` | query | 否 | `true` 时返回带标注框的图片(base64) |

**`options` 可用字段**(JSON 字符串,所有字段可选,省略则用服务端默认值):

```jsonc
{
  "engine": "vlm",               // 引擎: paddleocr(默认,本地 DB++) | vlm(Qwen-VL 视觉定位 OCR)
                                 //   vlm 时下方检测/识别参数无效(由模型自动决定);granularity 仍生效
  "text_det_thresh": 0.2,        // [paddleocr] 检测阈值,调低→召回更多框(艺术字救场)
  "text_det_box_thresh": 0.6,    // [paddleocr] 框内平均分阈值,调低→保留淡色字
  "text_det_unclip_ratio": 1.5,  // [paddleocr] 框放大系数,调大→框更松(艺术字笔画溢出)
  "text_rec_score_thresh": 0.5,  // [paddleocr] 识别置信度门槛
  "granularity": "paragraph",    // word=词框 / line=行框(默认) / paragraph=段落块(两引擎均支持)
  "paragraph_gap_ratio": 0.5,    // [paragraph] 行间距合并阈值
  "use_textline_orientation": true  // [paddleocr] 是否做文字行方向分类(旋转/倒置文字)
}
```

> **引擎选择**：`engine=vlm` 走 Qwen-VL 视觉定位 OCR,会把图中**所有**文字/艺术字/二维码/条码以归一化坐标框出并映射回像素坐标(大图自动分 tile,每 tile ≤ `OCR_VLM_OCR_MAX_SIDE`,避免大图请求卡死)。码检测同时跑 pyzbar 可靠解码,与 VLM 视觉框去重合并。需 `OCR_VLM_ENABLED=true` + `OCR_VLM_OCR_ENABLED=true` + API key,否则返回 503。


#### 响应

```jsonc
{
  "image_meta": {
    "width": 6483, "height": 5309,
    "tile_count": 16,
    "crop": [42, 38, 6442, 5238]   // 裁剪框,见 2.2
  },
  "items": [
    {
      "id": "t_001",
      "type": "text",                          // text | art_text | qr | barcode
      "text": "净含量:250ml",                    // 文字内容(qr/barcode 用 content 字段)
      "polygon": [[498,383],[659,383],[664,419],[504,444]],
      "bbox": [498, 383, 664, 444],
      "confidence": 0.97,
      "source": "paddleocr"                     // paddleocr | vlm | vlm_fallback | pyzbar
    }
  ],
  "options_used": { ... },
  "annotated_image_b64": null                   // annotate=true 时有值
}
```

#### `type` 类型说明

| `type` | 含义 | 内容字段 | 来源 |
|--------|------|----------|------|
| `text` | 普通文字 | `text` | `paddleocr` / `vlm` / `vlm_fallback` |
| `art_text` | 艺术字 | `text` | `vlm`（VLM 引擎可识别艺术字） |
| `qr` | 二维码 | `content` | `pyzbar` / `vlm` |
| `barcode` | 条码（含 69 码） | `content` | `pyzbar` / `vlm` |

> 提取内容时用 `item.text ?? item.content` 兼容两种类型。

#### 调用示例

```bash
# 基本识别
curl -F "url=http://内网图床/a.png" http://10.1.93.196:8000/analyze | jq

# 召回拉满(艺术字/淡色字救场)+ 段落粒度
curl -F "url=http://内网图床/a.png" \
  -F 'options={"text_det_thresh":0.2,"granularity":"paragraph"}' \
  http://10.1.93.196:8000/analyze | jq

# 只看文字+坐标(精简输出)
curl -F "url=http://..." http://10.1.93.196:8000/analyze \
  | jq '.items[] | {type, content: (.text // .content), bbox, confidence}'
```

---

### 3.2 `POST /understand` —— AI 整图理解

**不跑 OCR**,把整图交给 VLM 回答"这张图是什么"(分类、风格、关键元素)。

```bash
curl -F "url=http://内网图床/a.png" http://10.1.93.196:8000/understand | jq
```

---

### 3.3 `POST /agent/understand` —— AI Agent 深度理解

比 `/understand` 更深:Agent(Qwen3-max)通过 ReAct 循环调用 OCR/视觉工具,多步推理后给出结构化结论 + 推理过程。更准但更慢。

```bash
curl -F "url=http://内网图床/a.png" http://10.1.93.196:8000/agent/understand | jq
```

---

### 3.4 `POST /panels` —— 刀模图自动拆面

把刀模图按版面结构自动切成多个包装面(panel),返回每个面的 bbox 和裁剪图。

```bash
curl -F "url=http://内网图床/dieline.png" http://10.1.93.196:8000/panels | jq
# preview=true(默认)时返回每个 panel 的 base64 PNG 裁剪图
```

---

### 3.5 `POST /verify` —— 文案校验(合规检测)

对图片做 OCR 后,用**确定性规则**把识别出的文字与一份"标准文案"逐条比对,检测每条标准文案是否在图上出现(包装合规检测)。纯本地、无云模型、无需 API key。

#### 请求参数

| 参数 | 位置 | 必填 | 说明 |
|------|------|------|------|
| `file` / `url` | form | 二选一 | 图片输入(见 2.1) |
| `standard` | form | 是 | 标准文案,JSON 数组,每项见下表。空数组返回 400 |
| `options` | form | 否 | OCR 参数覆盖,JSON 字符串,同 `/analyze` |

**`standard` 数组每项字段**(所有字段除 `text` 外可选):

```jsonc
[
  {
    "text": "净含量450g",      // 要核对的文案(必填)
    "required": true,          // 必填(必须出现) / 选填(默认 true)
    "category": "净含量",       // 分组,任意字符串(可选)
    "id": "spec-12"            // 调用方 id,缺省自动补 v1/v2…(可选)
  }
]
```

#### 响应

```jsonc
{
  "image_meta": { "width": 6483, "height": 5309, "tile_count": 1 },
  "total": 3, "matched": 2, "partial": 0, "missing": 1,
  "pass": false,                // 所有 required 条目都 matched 才为 true
  "results": [
    {
      "entry_id": "v1",
      "text": "净含量450g",
      "required": true,
      "category": "净含量",
      "status": "matched",      // matched | partial | missing
      "similarity": 1.0,        // 0~1 召回率:标准文案有多少比例字符按序出现在 OCR 文本里
      "matched_item_ids": ["t_001", "t_002"],  // 命中的 OCR item(用于高亮,缺失时为 [])
      "matched_text": "净含量450g"             // 命中的归一化片段(便于人工核对)
    }
  ],
  "items": [ /* 完整的 OCR item 列表,同 /analyze,供前端渲染/高亮 */ ]
}
```

#### `status` 判定规则

两侧先归一化(NFKC 全角→半角 → 转小写 → 去掉空格/标点),再用 `difflib` 计算**召回率**(标准文案的字符有多少比例按顺序出现在 OCR 文本里):

| 召回率 | status | 含义 |
|--------|--------|------|
| ≥ `OCR_VERIFY_MATCH_THRESHOLD`(默认 0.85) | `matched` | 已出现 |
| ≥ `OCR_VERIFY_PARTIAL_THRESHOLD`(默认 0.60)且 < match | `partial` | 疑似有但存疑(OCR 错字 / 改写 / 真实差异,需人工复核) |
| < partial | `missing` | 未出现 |

```bash
curl -F "url=http://内网图床/a.png" \
     -F 'standard=[{"text":"净含量450g","required":true},{"text":"生产日期:见包装","required":false}]' \
     http://10.1.93.196:8000/verify | jq
```

---

### 3.6 `GET /tasks/{task_id}` —— 查询异步任务

查询 `/analyze` 异步任务的状态(见 2.3)。

```bash
curl http://10.1.93.196:8000/tasks/{task_id}
# {"task_id":"...","status":"done","result":{...}}
# status: pending → running → done / error
```

---

### 3.7 其它端点(刀模图交互工具,主要供 Web UI 用)

| 接口 | 用途 |
|------|------|
| `POST /panels/vlm` | VLM 辅助拆面(比 `/panels` 更准,需 VLM 配置) |
| `POST /panels/candidates` | 候选切割线检测(供交互式调整) |
| `POST /panels/compute` | 根据确认的切割线计算 panel 矩形 |

这些是刀模图交互编辑器配套接口,接入方一般不直接用,详见 `/docs`。

---

## 4. 错误处理

| HTTP 状态码 | 含义 | 处理建议 |
|-------------|------|----------|
| `200` | 同步处理成功 | 直接解析 `items` |
| `202` | 大图已接收,异步处理中 | 用 `task_id` 轮询 `/tasks/{id}` |
| `400` | 请求错误(未提供 file/url、URL 下载失败、超大小限制) | 检查 `detail` 字段,修正后重试 |
| `404` | task_id 不存在(异步任务) | 确认 task_id 正确 |
| `422` | 参数校验失败(options JSON 格式错误等) | 检查参数格式 |
| `500` | 服务内部错误 | 查服务日志,联系运维 |

错误响应体统一为:
```json
{"detail": "错误描述"}
```

---

## 5. 完整调用示例(Python)

```python
import json
import time
import requests

BASE = "http://10.1.93.196:8000"


def ocr(image_url: str = None, image_file: str = None,
        options: dict = None) -> dict:
    """调用 OCR 服务,自动处理同步/异步。"""
    data, files = {}, None
    if image_file:
        files = {"file": open(image_file, "rb")}
    elif image_url:
        data["url"] = image_url
    else:
        raise ValueError("需提供 image_url 或 image_file")
    if options:
        data["options"] = json.dumps(options)

    try:
        resp = requests.post(f"{BASE}/analyze", files=files, data=data, timeout=300)
        resp.raise_for_status()
        result = resp.json()
    finally:
        if files:
            files["file"].close()

    # 大图走异步,轮询取结果
    if resp.status_code == 202 and result.get("task_id"):
        return _poll_task(result["task_id"])
    return result


def _poll_task(task_id: str, interval: float = 5.0, timeout: float = 300.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        task = requests.get(f"{BASE}/tasks/{task_id}", timeout=30).json()
        if task["status"] == "done":
            return task["result"]
        if task["status"] == "error":
            raise RuntimeError(f"OCR 失败: {task.get('error')}")
        time.sleep(interval)
    raise TimeoutError(f"任务 {task_id} 超时")


# ── 使用 ──
result = ocr(image_url="http://内网图床/sample.png")
for item in result["items"]:
    text = item.get("text") or item.get("content")
    print(f"[{item['type']}] {text}  bbox={item['bbox']}")

# 召回拉满 + 段落粒度
result = ocr(image_url="http://...", options={
    "text_det_thresh": 0.2, "granularity": "paragraph",
})
```

---

## 6. 性能参考

基于 6483×5309 刀模图实测:

| 场景 | 耗时 | 说明 |
|------|------|------|
| 小图(<4000px)同步 | 1~10 秒 | 单 tile,首调用含模型加载 |
| 大图(>4000px)异步 | 100~200 秒 | 16 tile 串行 OCR + VLM 并发兜底 |
| 轮询间隔建议 | 5 秒 | 大图别频繁轮询 |

> 大图耗时主要在 PaddleOCR 的多 tile 串行推理。如需加速,可调整 `OCR_TILE_TARGET_SIZE`(增大→tile 更少)。

---

## 7. 配置项(运维侧)

接入方一般无需关心,列此供参考:

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `OCR_LARGE_IMAGE_THRESHOLD` | 4000 | 长边超过此值走异步 |
| `OCR_URL_FETCH_TIMEOUT` | 30 | URL 下载超时(秒) |
| `OCR_URL_FETCH_MAX_BYTES` | 104857600 | URL 下载大小上限(100MB) |
| `OCR_VLM_ENABLED` | false | 云 VLM 总开关（/understand、/agent、/panels/vlm、VLM OCR 引擎） |
| `OCR_VLM_OCR_FALLBACK_ENABLED` | false | `/analyze` 低置信 VLM 重认（需总开关 + key） |
| `OCR_VLM_OCR_ENABLED` | false | VLM OCR 引擎开关（`engine=vlm` 时需此项 + 总开关 + key） |
| `OCR_VLM_OCR_MODEL` | (空=复用 VLM_MODEL) | VLM OCR 视觉模型（建议 grounding 强的模型） |
| `OCR_VLM_OCR_MAX_SIDE` | 2048 | VLM OCR 每 tile 下采样长边（大图自动分 tile） |
| `OCR_OCR_ENGINE_DEFAULT` | paddleocr | 默认 OCR 引擎：paddleocr \| vlm |
| `OCR_VERIFY_MATCH_THRESHOLD` | 0.85 | `/verify` 召回率 ≥ 此值判为 matched |
| `OCR_VERIFY_PARTIAL_THRESHOLD` | 0.60 | `/verify` 召回率 ≥ 此值（且 < match）判为 partial |

完整配置见 `.env.example`。
