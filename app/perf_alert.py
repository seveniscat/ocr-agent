"""Performance alerting for /analyze — size-aware slow threshold + failures.

The pipeline already logs per-stage timings and the ``/logs`` page shows the
history, but nobody watches a log page at 2am. When a request runs unusually
long or fails, we push a text message to a Feishu group bot (transport in
``app/feishu.py``) so ops hears about it while users are still hurting.

Threshold shape (deliberate): OCR wall time scales with pixel count — tiling
turns a 4x-larger image into ~16x the PaddleOCR work — so a single fixed
"slow" number either spams on every large die-line or never fires on small
artwork. The threshold is therefore linear in megapixels::

    threshold_s = alert_slow_base_seconds + alert_slow_seconds_per_mp * (w*h / 1e6)

Defaults (30s + 5s/MP) put a 1000x800 upload at ~34s and a 4000x3000 sync
job at ~90s. Setting BOTH knobs to 0 disables slow alerts entirely (failure
alerts keep working; the URL still gates everything).

Design notes:
- **Judge here, send in main.py**: ``record_analyze`` returns the alert text
  or None; the caller submits it to the webhook thread pool. This module does
  no I/O, so calling it on the event loop is free, and tests need no mocking
  to exercise the decision logic.
- **Per-kind cooldown**: one giant job blowing the threshold shouldn't page
  ops once per tile-batch retry. Each alert kind ("slow" / "error") has its
  own cooldown window (default 5 min), so a slow spell followed by a failure
  still sends both.
- **Queue wait counts**: main.py measures end-to-end (request arrival →
  done/error), not just pipeline.run() — both OCR workers grinding a backlog
  IS the "卡顿" users feel, even when each individual run is fast.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional

logger = logging.getLogger(__name__)


def slow_threshold_seconds(megapixels: float, settings: Any) -> float:
    """Size-aware slow threshold: base + per-MP allowance."""
    return settings.alert_slow_base_seconds + settings.alert_slow_seconds_per_mp * megapixels


def _format_stages(stats_sink: Optional[dict]) -> str:
    """One-line per-stage breakdown from the pipeline's ``stats_sink``.

    Empty string when the sink has nothing (request failed before the pipeline
    ran) — no point printing a line of zeros.
    """
    if not stats_sink:
        return ""
    stages = (
        ("t_preprocess", "pre"),
        ("t_ocr", "ocr"),
        ("t_vlm", "vlm"),
        ("t_dedupe", "dedupe"),
        ("t_annotate", "annotate"),
    )
    parts = [
        f"{label}={(stats_sink.get(key) or 0.0):.1f}s" for key, label in stages
    ]
    # predict() is the slice of t_ocr spent inside PaddleOCR — the single
    # most common bottleneck, so call it out when it dominated.
    predict = stats_sink.get("t_ocr_predict") or 0.0
    if predict > 0.0:
        parts.append(f"predict={predict:.1f}s")
    return "分段: " + " ".join(parts)


class PerfAlerter:
    """Decides whether a finished /analyze call warrants a Feishu alert.

    One instance per process (module-level ``alerter`` in main.py); the
    cooldown map is guarded by a lock because completion callbacks arrive from
    both the event loop (sync path) and OCR worker threads (async path).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # kind ("slow" / "error") → time.monotonic() of the last sent alert.
        self._last_alert: dict[str, float] = {}

    def _cooldown_ok(self, kind: str, cooldown_s: float) -> bool:
        """Check + reserve the cooldown slot for ``kind``. Thread-safe."""
        now = time.monotonic()
        with self._lock:
            last = self._last_alert.get(kind)
            if last is not None and (now - last) < cooldown_s:
                return False
            self._last_alert[kind] = now
            return True

    def record_analyze(
        self,
        *,
        duration_s: float,
        status: str,
        settings: Any,
        error: Optional[str] = None,
        w: int = 0,
        h: int = 0,
        src: str = "none",
        task_id: Optional[str] = None,
        items: Optional[int] = None,
        stats_sink: Optional[dict] = None,
    ) -> Optional[str]:
        """Evaluate a finished /analyze call. Returns alert text, or None.

        ``status`` is ``"ok"`` or ``"error"`` (same values as the webhook and
        the /logs buffer). Never raises and never blocks on I/O — the caller
        decides how to deliver the returned text.
        """
        # Master gate: no bot configured → alerting is fully off.
        if not settings.feishu_webhook_url:
            return None

        path = "async" if task_id else "sync"
        where = f"尺寸 {w}x{h} | {path}"
        if task_id:
            where += f" task={task_id}"
        if src and src != "none":
            where += f" | 来源 {src}"

        if status == "error":
            if not settings.alert_on_error:
                return None
            if not self._cooldown_ok("error", settings.alert_cooldown_seconds):
                logger.info("feishu error alert suppressed by cooldown (%s)", error)
                return None
            lines = [
                "🚨 [ocr-agent] /analyze 失败",
                f"错误: {error or 'unknown'}",
                f"{where} | 耗时 {duration_s:.1f}s",
                time.strftime("%Y-%m-%d %H:%M:%S"),
            ]
            return "\n".join(lines)

        # Slow check. Both threshold knobs at 0 → threshold 0 → every request
        # would "exceed" it, so treat 0+0 as the off switch.
        mp = (w * h) / 1_000_000.0
        threshold = slow_threshold_seconds(mp, settings)
        allow = (
            settings.alert_slow_base_seconds + settings.alert_slow_seconds_per_mp
        )
        if allow <= 0 or duration_s <= threshold:
            return None
        if not self._cooldown_ok("slow", settings.alert_cooldown_seconds):
            logger.info(
                "feishu slow alert suppressed by cooldown (%.1fs > %.1fs)",
                duration_s, threshold,
            )
            return None

        header = (
            f"🐌 [ocr-agent] /analyze 慢请求\n"
            f"耗时 {duration_s:.1f}s > 阈值 {threshold:.1f}s"
            f"（base {settings.alert_slow_base_seconds:g}s"
            f" + {settings.alert_slow_seconds_per_mp:g}s/MP × {mp:.1f}MP）"
        )
        lines = [header, where]
        if items is not None:
            lines[-1] += f" | items={items}"
        stage_line = _format_stages(stats_sink)
        if stage_line:
            lines.append(stage_line)
        lines.append(time.strftime("%Y-%m-%d %H:%M:%S"))
        return "\n".join(lines)


# Process-wide alerter used by the /analyze hooks in main.py.
alerter = PerfAlerter()
