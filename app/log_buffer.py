"""In-memory ring buffer of recent OCR call statistics.

The pipeline already emits structured ``logger.info`` lines (per-stage timings,
VLM fallback counts, confidence drop counts). This module captures the same
numbers in a process-wide ring buffer so the Web UI can render a logs page
without parsing stderr.

Trade-offs (deliberate, matching the existing ``_tasks`` dict pattern):
  - In-memory only — a process restart clears the history.
  - Fixed capacity (500); oldest records age out automatically.
  - Thread-safe via a single lock; OCR runs in a 2-worker ThreadPoolExecutor.
"""

from __future__ import annotations

import itertools
import threading
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Optional

# How many recent calls to keep. 500 is plenty for interactive inspection
# while bounding memory; the Web UI defaults to fetching the last 200.
CAPACITY = 500

# Per-call cap on the fallback_crops detail list. The aggregate counts
# (vlm_sent / vlm_rescued / ...) are always complete; only the per-crop
# breakdown is truncated beyond this to bound memory on pathological inputs.
CAPACITY_CROPS = 200


@dataclass
class LogRecord:
    """One /analyze (or /understand, etc.) invocation, reduced to numbers.

    Fields map 1:1 to what the pipeline already logs, so a reader of stderr
    and a reader of this buffer see consistent data.
    """

    # --- request identity -----------------------------------------------
    ts: float                       # epoch seconds (time.time()); UI formats locally
    task_id: Optional[str]          # async-path task id; None on the sync path
    src: str                        # "url=..." or "file=<name>" or "none"
    engine: str                     # "paddleocr" / "vlm"
    status: str = "ok"              # "ok" / "error"
    error: Optional[str] = None     # short error string when status == "error"

    # --- image shape ----------------------------------------------------
    width: int = 0
    height: int = 0
    tiles: int = 0                  # grid tile count after preprocessing

    # --- item counts ----------------------------------------------------
    items_before: int = 0           # raw OCR output count
    items_after: int = 0            # final returned item count

    # --- per-stage wall time (seconds) ---------------------------------
    t_preprocess: float = 0.0
    t_ocr: float = 0.0
    t_ocr_predict: float = 0.0      # wall time inside PaddleOCR.predict() (sum across tiles)
    t_vlm: float = 0.0              # time spent in VLM fallback re-reads
    t_dedupe: float = 0.0
    t_annotate: float = 0.0         # annotated image rendering (0 if not requested)
    t_total: float = 0.0

    # --- OCR box-count breakdown (diagnostic for CPU-tuning) ------------
    # predict() is a black box (det+rec fused), so we can't split det vs rec
    # TIME, but the box COUNT ratio is the signal: det >> rec → detector noise
    # (raise det_thresh); high rec count × long t_ocr → rec bottleneck (tune
    # cpu_threads / rec_batch_size). All zero on the VLM engine path.
    ocr_predict_calls: int = 0       # number of predict() calls (= tile count)
    ocr_boxes_detected: int = 0      # detector output (dt_polys), includes rec-dropped
    ocr_boxes_recognized: int = 0    # recognizer output (rec_polys), passed threshold

    # --- VLM fallback (circular-text + low-confidence re-read) ---------
    # All zero when fallback didn't run (VLM disabled / everything confident).
    vlm_crops: int = 0              # number of crops actually sent to the VLM
    vlm_sent: int = 0               # alias kept for clarity in the UI
    vlm_rescued: int = 0            # VLM produced text where PaddleOCR was unsure
    vlm_empty: int = 0              # VLM returned nothing for the crop
    vlm_suspects: int = 0           # low-confidence text crops considered
    vlm_rings: int = 0              # circular / seal regions considered
    fallback_threshold: float = 0.0  # rec_confidence_fallback used this run
    # Per-crop breakdown for the /logs detail view: one dict per crop sent to
    # the VLM. Each has kind/box/orig_text/orig_conf/vlm_text/outcome. (The VLM
    # no longer supplies a usable score, so there is no vlm_conf field.)
    # Empty list when fallback didn't run; capped at CAPACITY_CROPS to bound
    # memory on pathological inputs (the aggregate counts above are unaffected).
    fallback_crops: list = field(default_factory=list)

    # --- confidence drop policy (/analyze only) ------------------------
    dropped: int = 0                # text boxes dropped for low confidence
    vlm_drop_threshold: float = 0.0  # rec_confidence_vlm_drop used this run
    # Per-rule breakdown of the universal _clean_items pass (all paths). Lets
    # the /logs UI show WHY items were dropped, not just how many.
    clean_empty: int = 0  # rule 1: empty/whitespace text
    clean_unrec: int = 0  # rule 2: recognized is False
    clean_short: int = 0  # rule 3: junk-short (< min_text_chars effective)
    clean_small: int = 0  # rule 4: small box (bbox min side < min_box_side)
    clean_low: int = 0    # rule 5: confidence < min_keep_confidence (off by default)

    # Stable record id, assigned by log_buffer_record. Lives last only because
    # dataclass field ordering requires defaults after non-defaults — it's the
    # key the /logs/crop image endpoint looks up (ts alone isn't unique when
    # calls arrive in the same millisecond).
    id: int = 0


_BUFFER: "deque[LogRecord]" = deque(maxlen=CAPACITY)
_LOCK = threading.Lock()


# ---------------------------------------------------------------------------
# VLM crop thumbnails — parallel image store for the /logs UI
# ---------------------------------------------------------------------------
# Raw JPEG thumbnails of the crops sent to the VLM (aligned with the record's
# fallback_crops detail list by index), keyed by LogRecord.id. Deliberately
# KEPT OUTSIDE LogRecord: asdict() feeds the /logs/api JSON response, and
# embedding hundreds of base64 images there would balloon every UI refresh.
# The UI fetches them lazily via GET /logs/crop/{rec_id}/{idx}.
#
# Bounded twice over: only records that sent crops register images, and the
# store FIFO-evicts oldest records first once the total byte budget is hit.
# Monotonic ids make oldest = smallest id.

# Total bytes of JPEG thumbnails retained across all records. A typical crop
# thumbnail is a few KB, so this comfortably holds the last few hundred
# records' images while capping worst-case memory.
IMAGES_MAX_BYTES = 64 * 1024 * 1024

_IMAGES: "deque[tuple[int, list[bytes]]]" = deque()
_IMAGES_BYTES = 0
_NEXT_ID = itertools.count(1)


def log_buffer_record(
    rec: LogRecord, crop_images: list | None = None
) -> None:
    """Append a record (and optionally its VLM crop thumbnails).

    Assigns the record's stable ``id``. ``crop_images`` must be a list of JPEG
    byte strings aligned with the record's ``fallback_crops`` entries; None /
    non-bytes entries (encode failures) are skipped so the UI's onerror hides
    just those thumbnails. Oldest records/images are evicted per the buffers'
    bounds.
    """
    global _IMAGES_BYTES
    with _LOCK:
        rec.id = next(_NEXT_ID)
        _BUFFER.append(rec)
        if crop_images:
            payload = [bytes(im) for im in crop_images if isinstance(im, (bytes, bytearray))]
            if payload:
                _IMAGES.append((rec.id, payload))
                _IMAGES_BYTES += sum(len(im) for im in payload)
                while _IMAGES_BYTES > IMAGES_MAX_BYTES and len(_IMAGES) > 1:
                    _dropped_id, dropped = _IMAGES.popleft()
                    _IMAGES_BYTES -= sum(len(im) for im in dropped)


def log_buffer_crop_image(rec_id: int, idx: int) -> bytes | None:
    """JPEG bytes of crop ``idx`` for record ``rec_id``, or None.

    Records are searched newest-first (ids are monotonic); a record whose
    images were byte-budget-evicted, a bad index, or an encode failure all
    return None — the UI treats that as "no thumbnail" and hides the image.
    """
    with _LOCK:
        for rid, images in reversed(_IMAGES):
            if rid == rec_id:
                if 0 <= idx < len(images):
                    return images[idx]
                return None
    return None


def log_buffer_snapshot(limit: int = 200) -> list[dict]:
    """Return up to ``limit`` most recent records, oldest→newest.

    Returns plain dicts (via dataclasses.asdict) so FastAPI can JSON-serialize
    directly; the UI re-sorts newest-first.
    """
    with _LOCK:
        items = list(_BUFFER)[-limit:]
    return [asdict(r) for r in items]


def log_buffer_count() -> int:
    """Current number of records held (for the UI header badge)."""
    with _LOCK:
        return len(_BUFFER)


def clear() -> None:
    """Drop all records and crop images (tests / manual reset; not in UI)."""
    global _IMAGES_BYTES
    with _LOCK:
        _BUFFER.clear()
        _IMAGES.clear()
        _IMAGES_BYTES = 0
