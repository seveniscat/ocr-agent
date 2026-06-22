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

from .config import Settings, get_settings
from .pipeline import Pipeline
from .schemas import AnalyzeResponse, PreprocessResponse, TaskStatus

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


@app.get("/healthz")
def healthz() -> dict:
    return {"status": "ok", "version": app.version}


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


def _settings() -> Settings:
    return get_settings()
