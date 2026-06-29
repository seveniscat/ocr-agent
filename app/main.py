"""FastAPI entry point.

Two endpoints + healthz:
- ``POST /analyze``: multipart upload. Small images return synchronously;
  large images return ``202`` with a ``task_id`` (polled via ``/tasks/{id}``).
- ``GET  /tasks/{id}``: poll an async task.

Run locally: ``uvicorn app.main:app --reload``
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import (
    BackgroundTasks,
    FastAPI,
    File,
    Form,
    HTTPException,
    Query,
    UploadFile,
)
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .config import Settings, get_settings
from .pipeline import Pipeline
from .schemas import (
    AnalyzeResponse,
    CandidatesResponse,
    ComputePanelsRequest,
    ComputePanelsResponse,
    PanelsResponse,
    TaskStatus,
    UnderstandingResult,
    VlmPanelsResponse,
)

app = FastAPI(title="ocr-agent", version="0.1.0")

logger = logging.getLogger(__name__)

# Wire the business loggers (app.*) into uvicorn's output. Without this their
# INFO/WARNING records are dropped (no handler attached) — which is why none of
# the pipeline/OCR/VLM timing logs showed up even though they were emitted.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

# Dedicated thread pool for large-image OCR. pipeline.run() is a heavy
# SYNCHRONOUS call (CPU + serial cloud VLM calls); running it on FastAPI's
# BackgroundTasks would block the single event-loop thread and stall EVERY
# request — including the trivial GET /tasks/{id} poll. Offloading to a real
# OS thread keeps the event loop free to answer polls while OCR grinds away.
# max_workers=2: OCR is CPU/memory-bound; more workers risk OOM and CPU
# contention on big die-line tiles. Bump if you have the headroom.
_ocr_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="ocr-worker")

# Dedicated thread pool for outbound webhook delivery. Kept separate from the
# OCR pool so a slow / hung callback endpoint can never starve an OCR worker —
# the webhook is fire-and-forget (deliver() never raises), and delivery latency
# must not gate detection throughput. 4 workers is enough for a handful of
# concurrent callbacks; bump if you fan out to many business systems at once.
_webhook_executor = ThreadPoolExecutor(
    max_workers=4, thread_name_prefix="webhook-worker"
)

# Serve the embedded web UI assets at /static/*.
_STATIC_DIR = Path(__file__).parent / "static"
if _STATIC_DIR.is_dir():
    app.mount("/static", StaticFiles(directory=str(_STATIC_DIR)), name="static")

# In-memory task store (v1; swap for Redis/DB in production).
_tasks: dict[str, TaskStatus] = {}


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Serve the built-in web UI (drag-drop upload + polygon overlay)."""
    return FileResponse(
        str(_STATIC_DIR / "index.html"),
        headers={"Cache-Control": "no-store, must-revalidate"},
    )


def _get_pipeline() -> Pipeline:
    """Lazy singleton — OCR model loads on first use, not import."""
    global _pipeline
    try:
        return _pipeline
    except NameError:
        _pipeline = Pipeline(get_settings())
        return _pipeline


def _get_pipeline_existing() -> "Pipeline | None":
    """Return the already-built pipeline, or None if not yet created.

    Unlike :func:`_get_pipeline` this never triggers lazy loading — used by the
    config-save path which needs to reset a stale VLM handle *if* one exists,
    without paying the cost of building the OCR pipeline just to save config.
    """
    try:
        return _pipeline  # noqa: F821 — defined by _get_pipeline on first use
    except NameError:
        return None


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "version": app.version}


async def _resolve_image(
    file: UploadFile | None, url: str | None
) -> bytes:
    """Resolve the image input to bytes, supporting both upload and URL.

    Every image endpoint accepts either a multipart ``file`` OR a ``url`` form
    field (file wins when both are given — backward compatible). The returned
    bytes feed straight into the existing pipeline, which already decodes bytes
    (``tiling.load_image`` / ``Pipeline.run`` / ``Image.open``).

    Raises HTTPException(400) when neither input is provided or the URL fetch
    fails (network error, non-2xx, size limit) — surfaced as a client error so
    callers can retry/fix the URL instead of seeing an opaque 500.
    """
    if file is not None:
        return await file.read()
    if url:
        from .fetch import FetchError, fetch_image

        s = _settings()
        try:
            return await fetch_image(
                url,
                timeout=s.url_fetch_timeout,
                max_bytes=s.url_fetch_max_bytes,
            )
        except FetchError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
    raise HTTPException(
        status_code=400, detail="provide an image via 'file' or 'url'"
    )


# ---------------------------------------------------------------------------
# VLM provider configuration (editable from the Web UI, persisted to .env)
# ---------------------------------------------------------------------------


class VLMConfigUpdate(BaseModel):
    """Editable VLM config fields. All optional — only provided fields persist.

    ``api_key`` is special: None/empty means "leave the stored key untouched",
    so the UI can save base_url/model changes without re-entering the secret.
    """

    base_url: str | None = None
    model: str | None = None
    api_key: str | None = None  # None/"" → don't touch the stored key
    vlm_enabled: bool | None = None
    vlm_ocr_fallback_enabled: bool | None = None
    understand_enabled: bool | None = None
    enable_thinking: bool | None = None


def _vlm_config_payload(s: "Settings") -> dict:
    """Shape echoed by both GET and POST — never includes the raw key."""
    from .envstore import mask

    return {
        "provider": s.vlm_provider,
        "base_url": s.vlm_base_url,
        "model": s.vlm_model,
        "api_key_masked": mask(s.vlm_api_key),
        "has_key": bool(s.vlm_api_key),
        "vlm_enabled": s.vlm_enabled,
        "vlm_ocr_fallback_enabled": s.vlm_ocr_fallback_enabled,
        "understand_enabled": s.understand_enabled,
        "enable_thinking": s.vlm_enable_thinking,
    }


@app.get("/config/vlm")
def get_vlm_config() -> JSONResponse:
    """Current VLM config with the API key masked (never the raw secret)."""
    return JSONResponse(status_code=200, content=_vlm_config_payload(_settings()))


@app.post("/config/vlm")
def save_vlm_config(body: VLMConfigUpdate) -> JSONResponse:
    """Persist VLM config to .env; takes effect immediately (no restart).

    Writes only the keys provided; an empty/None ``api_key`` leaves the stored
    key untouched so base_url/model edits don't force re-entering the secret.
    After writing, the settings cache and any cached VLM client are dropped so
    the next call rebuilds them with the new values.
    """
    from .envstore import upsert

    # Map model fields → .env keys. Only write what the caller actually sent.
    writes: list[tuple[str, str]] = []
    if body.base_url is not None:
        writes.append(("OCR_VLM_BASE_URL", body.base_url))
    if body.model is not None:
        writes.append(("OCR_VLM_MODEL", body.model))
    if body.api_key:  # empty string → don't overwrite the stored key
        writes.append(("OCR_VLM_API_KEY", body.api_key))
    if body.vlm_enabled is not None:
        writes.append(("OCR_VLM_ENABLED", str(body.vlm_enabled).lower()))
    if body.vlm_ocr_fallback_enabled is not None:
        writes.append(
            ("OCR_VLM_OCR_FALLBACK_ENABLED", str(body.vlm_ocr_fallback_enabled).lower())
        )
    if body.understand_enabled is not None:
        writes.append(("OCR_UNDERSTAND_ENABLED", str(body.understand_enabled).lower()))
    if body.enable_thinking is not None:
        writes.append(("OCR_VLM_ENABLE_THINKING", str(body.enable_thinking).lower()))

    for key, value in writes:
        upsert(key, value)

    # Drop caches so the next request rebuilds from the just-written .env.
    # ``cache_clear`` only exists on the real lru_cache wrapper; if get_settings
    # was swapped out (e.g. in tests) it's a plain function — clearing is a
    # no-op there because the test fixture rebuilds settings per call anyway.
    _clear = getattr(get_settings, "cache_clear", None)
    if callable(_clear):
        _clear()
    pipeline = _get_pipeline_existing()
    if pipeline is not None:
        # Refresh the pipeline's Settings snapshot too — otherwise a key saved
        # at runtime through the UI never reaches the VLM (the pipeline still
        # holds the Settings object captured at startup, before the key existed).
        pipeline.refresh_settings(get_settings())

    return JSONResponse(
        status_code=200, content=_vlm_config_payload(get_settings())
    )


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    background_tasks: BackgroundTasks,
    file: UploadFile | None = File(None, description="Image upload (mutually exclusive with 'url')."),
    url: str | None = Form(None, description="Image URL to download (used when 'file' is absent)."),
    annotate: bool = Query(False, description="Include annotated image in response"),
    options: str | None = Form(
        None,
        description='JSON string of OCROptions overrides, e.g. '
        '{"text_det_thresh":0.2,"granularity":"paragraph"}. '
        'All fields optional; omitted fields use server defaults.',
    ),
    callback_url: str | None = Form(
        None,
        description="Optional webhook URL. When set, the service POSTs a tiny "
                    "status payload (event/task_id/status/biz_id/timestamp) here "
                    "on completion or failure; the receiver then GETs "
                    "/tasks/{task_id} for the full result. Must be http(s).",
    ),
    callback_secret: str | None = Form(
        None,
        description="Optional shared secret for HMAC-SHA256 webhook signing. "
                    "When set, the POST carries X-Webhook-Signature: sha256=<hex> "
                    "over the raw body. Omit to send an unsigned callback.",
    ),
    biz_id: str | None = Form(
        None,
        description="Optional business id echoed verbatim in the webhook payload, "
                    "so the receiver can correlate the callback to its own record.",
    ),
) -> JSONResponse:
    settings = _settings()
    # Validate the callback URL early (before any OCR work) so a malformed URL
    # is a clean 400, not a job that runs and then fails to notify.
    if callback_url:
        _validate_callback_url(callback_url)

    # source label for logs: which input path the caller used.
    src = f"url={url}" if url else (f"file={file.filename}" if file else "none")
    t_start = time.perf_counter()
    data = await _resolve_image(file, url)

    # Parse optional OCR overrides.
    opt_obj = _parse_options(options)

    # Size gate: large images go async.
    from .tiling import image_size

    w, h = image_size(data)
    long_edge = max(w, h)

    if long_edge <= settings.large_image_threshold:
        # Synchronous path (small/medium images). Run in the thread pool so the
        # (synchronous, CPU-heavy) pipeline.run() doesn't block the event loop
        # either — even "small" images can take a couple seconds.
        pipeline = _get_pipeline()
        loop = asyncio.get_event_loop()
        resp = await loop.run_in_executor(
            _ocr_executor, lambda: pipeline.run(data, annotate=annotate, options=opt_obj)
        )
        logger.info(
            "/analyze sync %dx%d %s items=%d %.2fs",
            w, h, src, len(resp.items), time.perf_counter() - t_start,
        )

        # When a callback is requested, also publish the result under a task_id
        # so the receiver's GET /tasks/{id} works the same way as the async
        # path. The AnalyzeResponse already carries an optional task_id field,
        # so backfilling it is fully backward compatible (it stays None when no
        # callback is requested — the common, inline-return case is unchanged).
        if callback_url:
            sync_task_id = uuid.uuid4().hex
            resp.task_id = sync_task_id
            _tasks[sync_task_id] = TaskStatus(
                task_id=sync_task_id, status="done", result=resp
            )
            _fire_webhook(
                sync_task_id, "done",
                callback_url=callback_url,
                callback_secret=callback_secret,
                biz_id=biz_id,
            )
        return JSONResponse(status_code=200, content=resp.model_dump())

    # Async path (large images). Offload to the thread pool — NOT
    # BackgroundTasks — so the event loop stays free to answer /tasks polls
    # while OCR grinds. The 202 + task_id contract is unchanged.
    task_id = uuid.uuid4().hex
    _tasks[task_id] = TaskStatus(task_id=task_id, status="pending")

    def _run():
        t = time.perf_counter()
        try:
            _tasks[task_id].status = "running"
            pipeline = _get_pipeline()
            resp = pipeline.run(data, annotate=annotate, options=opt_obj)
            _tasks[task_id].result = resp
            _tasks[task_id].status = "done"
            logger.info(
                "/analyze async task=%s %dx%d %s items=%d done in %.2fs",
                task_id, w, h, src, len(resp.items), time.perf_counter() - t,
            )
            _fire_webhook(
                task_id, "done",
                callback_url=callback_url,
                callback_secret=callback_secret,
                biz_id=biz_id,
            )
        except Exception as exc:  # noqa: BLE001 — surface to caller
            err = f"{type(exc).__name__}: {exc}"
            _tasks[task_id].status = "error"
            _tasks[task_id].error = err
            logger.warning(
                "/analyze async task=%s failed in %.2fs: %s",
                task_id, time.perf_counter() - t, exc,
            )
            # Notify on failure too — the receiver otherwise can't tell a slow
            # job from a dead one and would poll forever.
            _fire_webhook(
                task_id, "error",
                callback_url=callback_url,
                callback_secret=callback_secret,
                biz_id=biz_id,
                error=err,
            )

    # Fire-and-forget on the pool; the response returns immediately with 202.
    loop = asyncio.get_event_loop()
    loop.run_in_executor(_ocr_executor, _run)
    logger.info(
        "/analyze async accepted task=%s %dx%d %s",
        task_id, w, h, src,
    )
    return JSONResponse(
        status_code=202,
        content=AnalyzeResponse(
            image_meta={"width": w, "height": h, "tile_count": 0},
            items=[],
            options_used=opt_obj,
            task_id=task_id,
        ).model_dump(),
    )


def _parse_options(raw: str | None) -> "OCROptions | None":
    """Parse the optional ``options`` form field into an OCROptions object."""
    if not raw:
        return None
    import json

    from .schemas import OCROptions

    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"options is not valid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError("options must be a JSON object")
    # Drop nulls so omitted fields fall back to server defaults.
    obj = {k: v for k, v in obj.items() if v is not None}
    return OCROptions(**obj) if obj else None


def _validate_callback_url(url: str) -> None:
    """Reject non-http(s) callback URLs.

    Callers are trusted internal systems (see app/fetch.py), so this is a guard
    against typos / accidental ``file://`` or ``ftp://`` rather than an SSRF
    defense. Surfaced as HTTP 400 so the caller can fix the URL immediately.
    """
    from urllib.parse import urlparse

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise HTTPException(
            status_code=400,
            detail="callback_url must be an absolute http(s) URL",
        )


def _fire_webhook(
    task_id: str,
    status: str,
    *,
    callback_url: str | None,
    callback_secret: str | None,
    biz_id: str | None,
    error: str | None = None,
) -> None:
    """Enqueue an outbound webhook delivery — fire-and-forget.

    No-op when ``callback_url`` is absent (the common case). Otherwise submit
    ``webhook.deliver`` to the webhook pool; it logs failures and never raises,
    so OCR throughput and the HTTP response are unaffected by callback delivery.
    """
    if not callback_url:
        return
    from .webhook import deliver

    _webhook_executor.submit(
        deliver,
        task_id,
        status,
        biz_id,
        callback_url,
        callback_secret,
        error,
    )


@app.get("/tasks/{task_id}", response_model=TaskStatus)
def get_task(task_id: str) -> JSONResponse:
    task = _tasks.get(task_id)
    if task is None:
        return JSONResponse(status_code=404, content={"detail": "task not found"})
    return JSONResponse(status_code=200, content=task.model_dump())


@app.post("/understand", response_model=UnderstandingResult)
async def understand(
    file: UploadFile | None = File(None, description="Image upload (mutually exclusive with 'url')."),
    url: str | None = Form(None, description="Image URL to download (used when 'file' is absent)."),
) -> JSONResponse:
    """AI-Native whole-image understanding: the VLM answers "what is this".

    A standalone branch that does NOT run OCR — it downscales the image to the
    VLM's sweet spot and asks one structured question. Always returns a parsed
    :class:`UnderstandingResult`; VLM/parse failures are surfaced via
    ``category_confidence=0`` + ``raw_note`` rather than an HTTP error.

    Reuses the OCR pipeline's lazily-initialized VLM handle (same provider/key),
    so enabling this costs no extra client setup. Level-1 = one whole-image
    call (synchronous, a few seconds); per-panel calls (level 2) are future work.
    """
    from .understanding import understand_image

    settings = _settings()
    if not settings.understand_enabled:
        return JSONResponse(
            status_code=503,
            content={"detail": "understanding layer disabled (OCR_UNDERSTAND_ENABLED=false)"},
        )
    if not settings.vlm_enabled:
        return JSONResponse(
            status_code=503,
            content={"detail": "VLM disabled (OCR_VLM_ENABLED=false); understanding needs a VLM"},
        )

    data = await _resolve_image(file, url)
    vlm = _get_understand_vlm()
    result = understand_image(data, vlm=vlm, settings=settings)
    return JSONResponse(status_code=200, content=result.model_dump())


@app.post("/agent/understand")
async def agent_understand(
    file: UploadFile | None = File(None, description="Image upload (mutually exclusive with 'url')."),
    url: str | None = Form(None, description="Image URL to download (used when 'file' is absent)."),
) -> JSONResponse:
    """AI-Native agent understanding: qwen3-max reasons + calls VLM/OCR tools.

    The brain (text-only) inspects the image through ``look``/``ocr_text``/
    ``describe`` tools over a ReAct loop, then emits a structured conclusion
    plus the full reasoning trace. Unlike the single-shot ``/understand``, the
    agent targets specific regions per question, so complex packaging images are
    understood accurately instead of guessed at once.

    Always returns an :class:`AgentResult`; on agent failure it degrades to an
    error conclusion (``fallback=True``) rather than 500.
    """
    from .agent.core import run_agent

    settings = _settings()
    if not settings.vlm_enabled:
        return JSONResponse(
            status_code=503,
            content={"detail": "VLM disabled (OCR_VLM_ENABLED=false); agent needs a VLM + reasoning LLM"},
        )

    data = await _resolve_image(file, url)
    result = run_agent(data, settings=settings, pipeline=_get_pipeline())
    return JSONResponse(status_code=200, content=result.model_dump())


def _get_understand_vlm():
    """Return a VLMProvider implementing ask_image(), reusing the OCR pipeline's
    lazily-loaded VLM so we don't spin up a second client.

    ``Pipeline._get_vlm()`` raises if the provider isn't ask_image-capable or
    the key is unset; we surface that as a 503-style message instead of a 500.
    """
    pipeline = _get_pipeline()
    try:
        return pipeline._get_vlm()  # noqa: SLF001 — reuse the lazy loader by design
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=503,
            detail=(
                f"VLM unavailable for understanding: {exc}. "
                "Set OCR_VLM_API_KEY / OCR_VLM_ENABLED."
            ),
        ) from exc


@app.post("/panels", response_model=PanelsResponse)
async def split_into_panels(
    file: UploadFile | None = File(None, description="Image upload (mutually exclusive with 'url')."),
    url: str | None = Form(None, description="Image URL to download (used when 'file' is absent)."),
    preview: bool = Query(
        True, description="Include base64 PNG crops of each panel.",
    ),
) -> JSONResponse:
    """Split a die-line image into its main box-face panels.

    Two-stage algorithm (reliable on finished design drafts, not just line art):
      1. ``detect_subject`` — binarize, dilate, take the LARGEST connected
         component as the six-face body. Verified correct on all samples.
      2. ``split_subject_into_panels`` — equal-subdivide the subject box by the
         standard unfolding layout (2×3 for ~square subjects, 1×4 for wide ones).

    The subject bbox is returned too so the UI can show it. Panels are an
    *initial* equal split; for non-equal faces the UI should let the user drag
    the cut lines (the existing ``/panels/candidates`` + ``/panels/compute``
    interactive flow supports that).
    """
    import io as _io

    from PIL import Image

    from .panels import split_panels_auto

    data = await _resolve_image(file, url)
    img = Image.open(_io.BytesIO(data))
    w, h = img.size

    panels, subject = split_panels_auto(img, return_images=preview)

    return JSONResponse(
        status_code=200,
        content=PanelsResponse(
            width=w,
            height=h,
            count=len(panels),
            subject=subject["bbox"] if subject else None,
            panels=[
                {
                    "index": i + 1,
                    "bbox": p.bbox,
                    "width": p.width,
                    "height": p.height,
                    "image_b64": p.image_b64,
                }
                for i, p in enumerate(panels)
            ],
        ).model_dump(),
    )


@app.post("/panels/vlm", response_model=VlmPanelsResponse)
async def split_panels_with_vlm(
    file: UploadFile | None = File(None, description="Image upload (mutually exclusive with 'url')."),
    url: str | None = Form(None, description="Image URL to download (used when 'file' is absent)."),
    preview: bool = Query(
        True, description="Include base64 PNG crops of each panel.",
    ),
    max_side: int | None = Query(
        None, ge=128, le=2048,
        description="Override the VLM image downscale long edge (default: OCR_PANELS_VLM_MAX_SIDE=512).",
    ),
) -> JSONResponse:
    """Detect die-line CUT LINES via the VLM and return them as editable lines.

    The VLM looks at a downscaled image (512px by default — cut lines are a
    low-detail task) and returns horizontal y-coords and vertical x-coords
    separating the box faces, in original-image pixel coords. The UI then lets
    the user drag each line into exact position before confirming via
    /panels/compute. ``lines`` (not ``panels``) carries the result; ``panels``
    is kept empty for response-model compatibility.

    Always returns 200 — VLM/parse failures surface as ``error`` + empty
    ``lines`` rather than an HTTP error.
    """
    from .understanding import detect_cut_lines

    settings = _settings()
    data = await _resolve_image(file, url)

    # Gate: VLM must be enabled and a key set.
    if not settings.vlm_enabled:
        return JSONResponse(
            status_code=200,
            content=VlmPanelsResponse(
                width=0, height=0, count=0, panels=[],
                error="VLM disabled (OCR_VLM_ENABLED=false)",
            ).model_dump(),
        )

    try:
        vlm = _get_understand_vlm()
    except Exception as exc:  # noqa: BLE001 — surfaced as error field, not 500
        return JSONResponse(
            status_code=200,
            content=VlmPanelsResponse(
                width=0, height=0, count=0, panels=[],
                error=f"VLM 不可用: {exc}",
            ).model_dump(),
        )

    result = detect_cut_lines(data, vlm=vlm, settings=settings, max_side=max_side)

    # Build the response. VlmPanelsResponse requires `panels`; we emit an empty
    # list (the UI consumes `lines`, added as an extra field below). Per-panel
    # PNG crops are produced by /panels/compute after the user confirms the cut
    # lines, so `preview` has no per-panel image to render at this stage.
    resp = VlmPanelsResponse(
        width=result["width"],
        height=result["height"],
        count=result["count"],
        panels=[],
        model=result.get("model", ""),
        error=result.get("error"),
    ).model_dump()
    resp["lines"] = result["lines"]
    return JSONResponse(status_code=200, content=resp)


@app.post("/panels/candidates", response_model=CandidatesResponse)
async def panel_candidates(
    file: UploadFile | None = File(None, description="Image upload (mutually exclusive with 'url')."),
    url: str | None = Form(None, description="Image URL to download (used when 'file' is absent)."),
    high_pct: float = Query(
        95.0, ge=50.0, le=99.9,
        description="Projection percentile for HIGH confidence (pre-selected). "
                    "Lower → more lines pre-selected.",
    ),
    mid_pct: float = Query(
        85.0, ge=50.0, le=99.9,
        description="Projection percentile for MEDIUM confidence.",
    ),
    low_pct: float = Query(
        70.0, ge=50.0, le=99.9,
        description="Projection percentile for LOW confidence.",
    ),
) -> JSONResponse:
    """Propose candidate cut lines for interactive panel splitting.

    Color-agnostic (gradient-based), so it works on both line-art die-lines and
    filled design drafts. Returns horizontal + vertical candidate lines with a
    confidence level; the UI lets the user toggle/add before computing panels.
    """
    import io as _io

    from PIL import Image

    from .panels import detect_candidate_lines

    data = await _resolve_image(file, url)
    img = Image.open(_io.BytesIO(data))
    res = detect_candidate_lines(
        img, high_pct=high_pct, mid_pct=mid_pct, low_pct=low_pct,
    )

    return JSONResponse(
        status_code=200,
        content=CandidatesResponse(
            width=res["width"],
            height=res["height"],
            h_lines=[
                {
                    "pos": ln["pos"],
                    "orientation": "h",
                    "confidence": ln["confidence"],
                    "selected": ln["selected"],
                }
                for ln in res["h_lines"]
            ],
            v_lines=[
                {
                    "pos": ln["pos"],
                    "orientation": "v",
                    "confidence": ln["confidence"],
                    "selected": ln["selected"],
                }
                for ln in res["v_lines"]
            ],
        ).model_dump(),
    )


@app.post("/panels/compute", response_model=ComputePanelsResponse)
async def compute_panels_endpoint(
    h_lines: str = Form("[]", description="JSON array of confirmed y-coords"),
    v_lines: str = Form("[]", description="JSON array of confirmed x-coords"),
    width: int = Form(...),
    height: int = Form(...),
    file: UploadFile | None = File(
        None, description="Optional image; when present each panel gets a PNG crop."
    ),
    url: str | None = Form(
        None, description="Optional image URL; used for PNG crops when 'file' is absent."
    ),
) -> JSONResponse:
    """Compute panel rectangles from user-confirmed cut lines.

    Pure geometry (no image processing). When ``file`` or ``url`` is supplied,
    each panel includes a base64 PNG crop; otherwise only bboxes are returned.
    ``h_lines`` and ``v_lines`` are JSON-encoded int arrays (the UI sends them
    as form fields alongside the optional image upload).
    """
    import base64
    import io as _io
    import json

    from PIL import Image

    from .panels import compute_panels

    hs = json.loads(h_lines) if h_lines else []
    vs = json.loads(v_lines) if v_lines else []
    panels = compute_panels(hs, vs, width, height)

    # Optional: render PNG crops for each panel from the source image. Accepts
    # either a multipart file or a URL (file wins); both absent → bbox-only,
    # matching the original optional-file behavior.
    source_img = None
    if file is not None or url:
        try:
            data = await _resolve_image(file, url)
            source_img = Image.open(_io.BytesIO(data)).convert("RGB")
        except Exception:  # noqa: BLE001 — best-effort crop
            source_img = None

    out_panels = []
    for i, p in enumerate(panels):
        img_b64 = None
        if source_img is not None:
            buf = _io.BytesIO()
            source_img.crop(tuple(p.bbox)).save(buf, format="PNG")
            img_b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        out_panels.append(
            {
                "index": i + 1,
                "bbox": p.bbox,
                "width": p.width,
                "height": p.height,
                "image_b64": img_b64,
            }
        )

    return JSONResponse(
        status_code=200,
        content=ComputePanelsResponse(
            width=width,
            height=height,
            count=len(out_panels),
            panels=out_panels,
        ).model_dump(),
    )


def _settings() -> Settings:
    return get_settings()
