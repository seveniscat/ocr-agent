"""FastAPI entry point.

Two endpoints + healthz:
- ``POST /analyze``: multipart upload. Small images return synchronously;
  large images return ``202`` with a ``task_id`` (polled via ``/tasks/{id}``).
- ``GET  /tasks/{id}``: poll an async task.

Run locally: ``uvicorn app.main:app --reload``
"""
from __future__ import annotations

import uuid
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, Form, Query, UploadFile
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
    PreprocessResponse,
    TaskStatus,
    UnderstandingResult,
    VlmPanelsResponse,
)

app = FastAPI(title="ocr-agent", version="0.1.0")

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
    file: UploadFile = File(...),
    annotate: bool = Query(False, description="Include annotated image in response"),
    options: str | None = Form(
        None,
        description='JSON string of OCROptions overrides, e.g. '
        '{"text_det_thresh":0.2,"granularity":"paragraph"}. '
        'All fields optional; omitted fields use server defaults.',
    ),
) -> JSONResponse:
    settings = _settings()
    data = await file.read()

    # Parse optional OCR overrides.
    opt_obj = _parse_options(options)

    # Size gate: large images go async.
    from .tiling import image_size

    w, h = image_size(data)
    long_edge = max(w, h)

    if long_edge <= settings.large_image_threshold:
        # Synchronous path (small/medium images).
        pipeline = _get_pipeline()
        resp = pipeline.run(data, annotate=annotate, options=opt_obj)
        return JSONResponse(status_code=200, content=resp.model_dump())

    # Async path (large images).
    task_id = uuid.uuid4().hex
    _tasks[task_id] = TaskStatus(task_id=task_id, status="pending")

    def _run():
        try:
            _tasks[task_id].status = "running"
            pipeline = _get_pipeline()
            resp = pipeline.run(data, annotate=annotate, options=opt_obj)
            _tasks[task_id].result = resp
            _tasks[task_id].status = "done"
        except Exception as exc:  # noqa: BLE001 — surface to caller
            _tasks[task_id].status = "error"
            _tasks[task_id].error = f"{type(exc).__name__}: {exc}"

    background_tasks.add_task(_run)
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


@app.post("/preprocess", response_model=PreprocessResponse)
async def preprocess(
    file: UploadFile = File(...),
    threshold: int = Query(
        240, ge=0, le=255,
        description="Grayscale cutoff for 'ink' pixels; < this = content.",
    ),
    padding: int = Query(
        0, ge=0, le=500,
        description="Extra margin (px) kept on every side after cropping.",
    ),
) -> JSONResponse:
    """Preview the die-line auto-crop WITHOUT running OCR.

    Returns the original size, the crop box (original-image coords), and a
    base64 PNG of the cropped region so the UI can show a side-by-side.
    """
    import base64
    import io

    from PIL import Image

    from .preprocess import autocrop

    data = await file.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    import numpy as np

    arr = np.array(img)
    h, w = arr.shape[:2]

    cropped, crop_box = autocrop(arr, threshold=threshold, padding=padding)

    removed = 0.0
    cw, ch = w, h
    cropped_b64 = None
    if crop_box is not None:
        ch, cw = cropped.shape[:2]
        removed = 1.0 - (cw * ch) / (w * h)
        buf = io.BytesIO()
        Image.fromarray(cropped).save(buf, format="PNG")
        cropped_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

    return JSONResponse(
        status_code=200,
        content=PreprocessResponse(
            width=w,
            height=h,
            crop=crop_box,
            cropped_width=cw if crop_box else None,
            cropped_height=ch if crop_box else None,
            cropped_image_b64=cropped_b64,
            removed_ratio=removed,
        ).model_dump(),
    )


@app.get("/tasks/{task_id}", response_model=TaskStatus)
def get_task(task_id: str) -> JSONResponse:
    task = _tasks.get(task_id)
    if task is None:
        return JSONResponse(status_code=404, content={"detail": "task not found"})
    return JSONResponse(status_code=200, content=task.model_dump())


@app.post("/understand", response_model=UnderstandingResult)
async def understand(file: UploadFile = File(...)) -> JSONResponse:
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

    data = await file.read()
    vlm = _get_understand_vlm()
    result = understand_image(data, vlm=vlm, settings=settings)
    return JSONResponse(status_code=200, content=result.model_dump())


def _get_understand_vlm():
    """Return a VLMProvider implementing ask_image(), reusing the OCR pipeline's
    lazily-loaded VLM so we don't spin up a second client.

    ``Pipeline._get_vlm()`` raises if the provider isn't ask_image-capable or
    the key is unset; we surface that as a 503-style message instead of a 500.
    """
    from fastapi import HTTPException

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
    file: UploadFile = File(...),
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

    data = await file.read()
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
    file: UploadFile = File(...),
    preview: bool = Query(
        True, description="Include base64 PNG crops of each panel.",
    ),
) -> JSONResponse:
    """Split a die-line into faces using the VLM (semantic layout recognition).

    Unlike the geometry-based ``/panels`` (which fails on finished design
    drafts), this asks the VLM to identify each box face directly. The VLM
    returns normalized bboxes; we scale them back to original-image pixels and,
    optionally, render PNG crops.

    Always returns 200 — VLM/parse failures surface as ``error`` + empty
    ``panels`` rather than an HTTP error, so the UI can show a message.
    """
    import io as _io

    from PIL import Image

    from .understanding import split_panels_vlm

    settings = _settings()
    data = await file.read()

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

    result = split_panels_vlm(data, vlm=vlm, settings=settings)

    # Optionally render PNG crops from the original image.
    if preview and result["panels"]:
        import base64

        try:
            src = Image.open(_io.BytesIO(data)).convert("RGB")
            for p in result["panels"]:
                buf = _io.BytesIO()
                src.crop(tuple(p["bbox"])).save(buf, format="PNG")
                p["image_b64"] = base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception:  # noqa: BLE001 — crops are best-effort
            pass

    return JSONResponse(
        status_code=200,
        content=VlmPanelsResponse(
            width=result["width"],
            height=result["height"],
            count=result["count"],
            panels=result["panels"],
            model=result.get("model", ""),
            error=result.get("error"),
        ).model_dump(),
    )


@app.post("/panels/candidates", response_model=CandidatesResponse)
async def panel_candidates(
    file: UploadFile = File(...),
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

    data = await file.read()
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
) -> JSONResponse:
    """Compute panel rectangles from user-confirmed cut lines.

    Pure geometry (no image processing). When ``file`` is supplied, each panel
    includes a base64 PNG crop; otherwise only bboxes are returned. ``h_lines``
    and ``v_lines`` are JSON-encoded int arrays (the UI sends them as form
    fields alongside the optional image upload).
    """
    import base64
    import io as _io
    import json

    from PIL import Image

    from .panels import compute_panels

    hs = json.loads(h_lines) if h_lines else []
    vs = json.loads(v_lines) if v_lines else []
    panels = compute_panels(hs, vs, width, height)

    # Optional: render PNG crops for each panel from the uploaded source image.
    source_img = None
    if file is not None:
        data = await file.read()
        try:
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
