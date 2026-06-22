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

from fastapi import BackgroundTasks, FastAPI, File, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import Settings, get_settings
from .pipeline import Pipeline
from .schemas import AnalyzeResponse, TaskStatus

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
    return FileResponse(str(_STATIC_DIR / "index.html"))


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
) -> JSONResponse:
    settings = _settings()
    data = await file.read()

    # Size gate: large images go async.
    from .tiling import image_size

    w, h = image_size(data)
    long_edge = max(w, h)

    if long_edge <= settings.large_image_threshold:
        # Synchronous path (small/medium images).
        pipeline = _get_pipeline()
        resp = pipeline.run(data, annotate=annotate)
        return JSONResponse(status_code=200, content=resp.model_dump())

    # Async path (large images).
    task_id = uuid.uuid4().hex
    _tasks[task_id] = TaskStatus(task_id=task_id, status="pending")

    def _run():
        try:
            _tasks[task_id].status = "running"
            pipeline = _get_pipeline()
            resp = pipeline.run(data, annotate=annotate)
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
            task_id=task_id,
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
