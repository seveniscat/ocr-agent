"""Pipeline orchestration.

Flow (v1 scope: long edge ≤ 4000px):
    load image → [optional autocrop]
    → if long edge ≤ 4000: single PaddleOCR.predict() on full image
    → else: tile grid (future / >4000 async path)
    → optional paragraph merge → optional VLM fallback → dedupe → return

The pipeline is a thin coordinator: each stage lives in its own module so it
can be swapped or unit-tested independently.
"""
from __future__ import annotations

import base64
import logging
import time

from .config import Settings
from .schemas import AnalyzeResponse, ImageMeta, Item, OCROptions
from .tiling import (
    GridSpec,
    crop_tile,
    dedupe_items,
    load_image,
    offset_polygon,
    plan_grid,
    polygon_to_bbox,
    renumber,
    tile_specs,
)

logger = logging.getLogger(__name__)

# Prompt for the circular-region VLM read. A whole ring (logo/seal/badge text on
# an arc) is cropped as its bounding box and sent with this prompt — the VLM is
# asked to read the characters around the ring. We do NOT polar-unroll here
# (path A): rely on the VLM's understanding of arc layout. If this proves
# unreliable on dense rings, the upgrade path is cv2.warpPolar before sending.
_CIRCULAR_PROMPT = (
    "这是包装上沿圆弧/环形排布的文字（圆形 logo、印章、徽章上的弧形文字）。"
    "请按顺时针方向，从顶部（12 点钟方向）开始，原样读出环上所有可见文字。"
    "用 / 分隔各段弧形文字（如顶部和底部是两段）。"
    "只输出文字本身，不要解释、不要引号。如果该区域没有文字，只输出: EMPTY"
)


class Pipeline:
    """Stateful coordinator holding lazily-loaded model handles."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        # Lazy: these load heavy models on first use.
        self._ocr = None
        self._codes = None
        self._codes_unavailable = False  # set True once we know the lib is missing
        self._vlm = None

    # -- lazy loaders -------------------------------------------------------

    def _get_ocr(self):
        if self._ocr is None:
            from .ocr.detector import OCREngine

            self._ocr = OCREngine(self.settings)
        return self._ocr

    def _get_codes(self):
        if self._codes is None:
            from .codes.qrcode import CodeEngine

            self._codes = CodeEngine()
        return self._codes

    def _get_codes_or_none(self):
        """Return the code engine, or None if the zbar lib is unavailable.

        The first failure probes the lib once and remembers the result so we
        don't spam the log per request. Code detection is a *secondary* channel
        — its absence must never break the primary OCR flow.
        """
        if self._codes_unavailable:
            return None
        engine = self._get_codes()
        if engine.available():
            return engine
        logger.warning(
            "Code engine unavailable (zbar lib missing); QR/barcode channel "
            "disabled. Fix: brew install zbar (macOS) / apt-get install libzbar0."
        )
        self._codes_unavailable = True
        return None

    def _get_vlm(self):
        if self._vlm is None:
            from .vlm.base import build_vlm

            self._vlm = build_vlm(self.settings)
        return self._vlm

    def refresh_settings(self, settings: "Settings") -> None:
        """Hot-swap the Settings reference after the user edited config at runtime.

        Called by the ``POST /config/vlm`` endpoint so a key saved through the
        Web UI takes effect immediately, without a restart and without
        rebuilding the (expensive) OCR engine. The cached VLM client is dropped
        so the next ``_get_vlm()`` rebuilds it with the new key/base_url/model.

        Note: the OCR engine captures settings at construction too, but only VLM
        settings are editable through the UI, so we intentionally leave
        ``self._ocr`` untouched.
        """
        self.settings = settings
        self._vlm = None

    # -- main entry ---------------------------------------------------------

    def run(
        self,
        image_data: bytes,
        annotate: bool = False,
        options: "OCROptions | None" = None,
        image_url: str | None = None,
        confidence_policy: bool = False,
        stats_sink: dict | None = None,
    ) -> AnalyzeResponse:
        # ``stats_sink`` (when provided by the caller) collects the same numbers
        # already emitted via logger.info, so the /logs Web UI can render a
        # structured history without parsing stderr. Caller owns the dict; we
        # only write to it. See app/log_buffer.py for the consumer.
        t0 = time.perf_counter()
        img = load_image(image_data)
        h, w = img.shape[:2]
        orig_w, orig_h = w, h  # before autocrop, for the log

        # --- Preprocess: crop blank margins around the die-line artwork. ---
        # The cropped image becomes the working image for everything below, so
        # all item coordinates are naturally in cropped space; the one `crop`
        # offset is echoed back in image_meta for callers to remap to the
        # original. Blank image → no crop, fall through unchanged.
        t_pre = time.perf_counter()
        crop_box = None
        if self.settings.preprocess_autocrop:
            from .preprocess import autocrop

            img, crop_box = autocrop(
                img,
                threshold=self.settings.preprocess_autocrop_threshold,
                padding=self.settings.preprocess_autocrop_padding,
            )
            h, w = img.shape[:2]
        t_pre = time.perf_counter() - t_pre

        max_side = max(w, h)

        # Resolve which engine to run. Per-request override wins; else server
        # default. ``vlm`` routes through Qwen-VL grounding OCR (peer of the
        # PaddleOCR path below); everything else is the local DB++ det+rec.
        engine = "paddleocr"
        if options is not None and options.engine:
            engine = options.engine
        elif self.settings.ocr_engine_default:
            engine = self.settings.ocr_engine_default

        all_items: list[Item] = []
        t_ocr = time.perf_counter()

        if engine == "vlm":
            all_items, grid = self._run_vlm_engine(img, image_url=image_url)
        else:
            all_items, grid = self._run_paddle_engine(img, w, h, max_side, options)
        t_ocr = time.perf_counter() - t_ocr
        n_after_ocr = len(all_items)

        # Pull per-tile OCR stats from the engine (boxes_detected / recognized,
        # predict wall time). Empty when the VLM engine ran or engine isn't
        # loaded yet — `.get()` keeps the sink clean in those cases.
        ocr_engine_stats = {}
        if engine != "vlm" and self._ocr is not None:
            ocr_engine_stats = self._ocr.ocr_stats

        # Re-derive granularity + paragraph params from the request so the
        # shared post-processing (paragraph merge etc.) applies to BOTH engines.
        if options is not None:
            gran = options.granularity
            gap = options.paragraph_gap_ratio
            xov = options.paragraph_x_overlap
        else:
            gran, gap, xov = None, None, None
        effective_gran = gran or self.settings.ocr_granularity

        # --- merge same-line overlaps (mixed-script detection splits) BEFORE
        # dedupe. The detector often splits one line into two boxes when it
        # mixes scripts (e.g. English + Korean); after unclipping those overlap
        # in x. dedupe won't fold them (different text → low similarity), so we
        # merge them here so each pixel ends up in at most one box. ---
        from .tiling import merge_same_line_overlaps

        all_items = merge_same_line_overlaps(
            all_items,
            x_overlap_ratio=self.settings.same_line_merge_x_overlap,
        )

        # --- dedupe at line level before paragraph merge (tile-seam duplicates
        # break geometric grouping if left in place). ---
        t_dedup = time.perf_counter()
        merge_x = self.settings.tile_merge_x_thres
        merge_y = self.settings.tile_merge_y_thres
        all_items = dedupe_items(
            all_items,
            merge_x_thres=merge_x,
            merge_y_thres=merge_y,
        )
        if effective_gran == "paragraph":
            from .ocr.aggregator import apply_paragraph_granularity

            n_lines = sum(
                1 for it in all_items if it.type == "text" and it.source == "paddleocr"
            )
            all_items = apply_paragraph_granularity(
                all_items,
                gap_ratio=gap
                if gap is not None
                else self.settings.ocr_paragraph_gap_ratio,
                x_overlap=xov
                if xov is not None
                else self.settings.ocr_paragraph_x_overlap,
            )
            n_blocks = sum(
                1
                for it in all_items
                if it.granularity == "paragraph" and it.type == "text"
            )
            logger.info(
                "paragraph merge: %d lines -> %d blocks (gap=%.2f x_ov=%.2f)",
                n_lines,
                n_blocks,
                gap if gap is not None else self.settings.ocr_paragraph_gap_ratio,
                xov if xov is not None else self.settings.ocr_paragraph_x_overlap,
            )

        # --- optional VLM fallback (opt-in; PaddleOCR is the default OCR path) ---
        t_vlm, vlm_calls = time.perf_counter(), 0
        all_items, vlm_calls, vlm_stats = self._maybe_vlm_fallback(img, all_items)
        t_vlm = time.perf_counter() - t_vlm

        # --- final dedupe (paragraph blocks can still overlap at tile seams) ---
        all_items = dedupe_items(
            all_items,
            merge_x_thres=merge_x,
            merge_y_thres=merge_y,
        )

        # --- confidence policy (POST /analyze only): drop text boxes whose FINAL
        # confidence is still below rec_confidence_drop after the VLM fallback
        # pass. VLM re-read happens above, so a box the VLM rescued above the
        # threshold survives; one that stayed low is discarded. Only text items
        # are dropped — qr/barcode confidence has different semantics and those
        # codes are valuable either way. /verify opts out (confidence_policy=False)
        # because it needs every OCR'd character to match standard copy. ---
        n_dropped = 0
        drop_threshold_used = 0.0
        if confidence_policy:
            n_before_drop = len(all_items)
            all_items = self._drop_low_confidence(all_items)
            n_dropped = n_before_drop - len(all_items)
            # Mirror the clamp in _drop_low_confidence so the UI shows the
            # actually-applied threshold (not the raw setting).
            drop_threshold_used = min(
                self.settings.rec_confidence_drop,
                self.settings.rec_confidence_fallback,
            )

        all_items = renumber(all_items, prefix="t")
        t_dedup = time.perf_counter() - t_dedup

        # --- per-stage timing log. vlm_calls is the number of CROPS inspected;
        # they're sent in ~⌈crops/batch⌉ batched requests (each ~20 crops in
        # one multi-image call), so wall-clock no longer scales linearly with it.
        # n_dropped is only nonzero on the /analyze path (confidence_policy).
        # boxes=det/rec is the detector-vs-recognizer box count (cumulative
        # across tiles): det >> rec → detector noise (raise det_thresh); high
        # rec count × long ocr → rec bottleneck (tune cpu_threads / rec_batch). ---
        t_total = time.perf_counter() - t0
        _bd = ocr_engine_stats.get("boxes_detected", 0)
        _br = ocr_engine_stats.get("boxes_recognized", 0)
        logger.info(
            "pipeline.run: %dx%d→%dx%d tiles=%d items=%d→%d "
            "preprocess=%.2fs ocr=%.2fs boxes=det/rec=%d/%d "
            "vlm(crops=%d)=%.2fs dedupe=%.2fs drop=%d total=%.2fs",
            orig_w, orig_h, w, h, grid.count, n_after_ocr, len(all_items),
            t_pre, t_ocr, _bd, _br, vlm_calls, t_vlm, t_dedup, n_dropped, t_total,
        )

        response = AnalyzeResponse(
            image_meta=ImageMeta(
                width=w, height=h, tile_count=grid.count, crop=crop_box
            ),
            items=all_items,
            options_used=options,
        )

        t_annot = 0.0
        if annotate:
            from .viz.annotator import annotate_image

            t_annot = time.perf_counter()
            annotated = annotate_image(img, all_items, self.settings)
            response.annotated_image_b64 = _b64_png(annotated)
            t_annot = time.perf_counter() - t_annot
            logger.info("pipeline.run: annotate=%.2fs", t_annot)

        # Publish the structured run stats for the /logs Web UI. Keys mirror
        # LogRecord fields in app/log_buffer.py. vlm_stats is {} when fallback
        # didn't run (VLM disabled / nothing suspect), so the .get() defaults
        # carry through cleanly.
        if stats_sink is not None:
            stats_sink.update({
                "tiles": grid.count,
                "items_before": n_after_ocr,
                "items_after": len(all_items),
                "t_preprocess": t_pre,
                "t_ocr": t_ocr,
                "t_ocr_predict": ocr_engine_stats.get("t_predict", 0.0),
                "ocr_predict_calls": ocr_engine_stats.get("predict_calls", 0),
                "ocr_boxes_detected": ocr_engine_stats.get("boxes_detected", 0),
                "ocr_boxes_recognized": ocr_engine_stats.get("boxes_recognized", 0),
                "t_vlm": t_vlm,
                "t_dedupe": t_dedup,
                "t_annotate": t_annot,
                "t_total": t_total,
                "vlm_crops": vlm_calls,
                "vlm_sent": vlm_stats.get("sent", 0),
                "vlm_rescued": vlm_stats.get("rescued", 0),
                "vlm_empty": vlm_stats.get("empty", 0),
                "vlm_suspects": vlm_stats.get("suspects", 0),
                "vlm_rings": vlm_stats.get("rings", 0),
                "fallback_threshold": vlm_stats.get("threshold", 0.0),
                "fallback_crops": vlm_stats.get("crops", []),
                "dropped": n_dropped,
                "drop_threshold": drop_threshold_used,
            })

        return response

    # -- helpers ------------------------------------------------------------

    def _run_vlm_engine(
        self, img, image_url: str | None = None
    ) -> tuple[list[Item], "GridSpec"]:
        """Run the Qwen-VL grounding OCR engine over the (autocropped) image.

        Returns ``(items, grid)`` where ``grid`` is the tiling plan (for the
        ``image_meta.tile_count`` echo). Raises a clear error if VLM OCR is not
        configured — the caller surfaces it.

        When ``image_url`` is given, the VLM receives the public URL directly
        (no tiling, no base64) — the preferred path for large images and the
        one matching the proven calling convention. Otherwise the image is
        tiled + base64'd per tile (fallback for multipart file uploads).
        """
        if not self.settings.vlm_enabled:
            raise RuntimeError(
                "VLM OCR requires OCR_VLM_ENABLED=true (and OCR_VLM_OCR_ENABLED)."
            )
        if not self.settings.vlm_ocr_enabled:
            raise RuntimeError(
                "VLM OCR engine is disabled (OCR_VLM_OCR_ENABLED=false). "
                "Set it true to use engine=vlm."
            )
        try:
            vlm = self._get_vlm()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"VLM unavailable for OCR: {exc}. "
                "Set OCR_VLM_API_KEY / OCR_VLM_ENABLED."
            ) from exc

        from .vlm_ocr import run_vlm_ocr

        items = run_vlm_ocr(img, vlm, self.settings, image_url=image_url)

        # Re-derive the grid plan (same params run_vlm_ocr used) only for the
        # tile_count echo — cheap, and keeps the response shape consistent.
        h, w = img.shape[:2]
        target = (
            max(w, h)
            if max(w, h) <= self.settings.vlm_ocr_max_side
            else self.settings.vlm_ocr_max_side
        )
        grid = plan_grid(
            w, h, target_size=target, overlap=self.settings.tile_overlap
        )
        return items, grid

    def _run_paddle_engine(
        self, img, w: int, h: int, max_side: int, options
    ) -> tuple[list[Item], "GridSpec"]:
        """Run the local PaddleOCR engine (per-tile det+rec + pyzbar codes).

        Returns ``(items, grid)`` — same shape as :meth:`_run_vlm_engine` so the
        shared post-processing in :meth:`run` is engine-neutral.
        """
        # ≤ small_image_threshold (default 4000): one tile → official predict() path.
        target_size = (
            max_side
            if max_side <= self.settings.small_image_threshold
            else self.settings.tile_target_size
        )
        grid = plan_grid(
            w, h,
            target_size=target_size,
            overlap=self.settings.tile_overlap,
        )
        specs = tile_specs(grid)

        all_items: list[Item] = []

        ocr = self._get_ocr()
        # Reset per-request stats so the counters reflect THIS run only (the
        # engine is a singleton reused across requests).
        ocr.reset_stats()
        codes = self._get_codes_or_none()  # may be None (lib missing) → skip

        # Translate per-request OCR overrides to detector call args.
        if options is not None:
            predict_kwargs = options.to_predict_kwargs()
        else:
            predict_kwargs = {}

        # OCR always emits lines; paragraph merge runs globally after all tiles.
        ocr_gran = "line"

        for spec in specs:
            tile = crop_tile(img, spec)

            # --- text (primary channel; errors here are fatal) ---
            for det in ocr.detect_and_recognize(
                tile,
                predict_kwargs=predict_kwargs,
                granularity=ocr_gran,
            ):
                global_poly = offset_polygon(det.polygon, spec.x0, spec.y0)
                # if paragraph mode, offset the per-line quads too
                global_lines = None
                if det.lines:
                    global_lines = [
                        offset_polygon(ln, spec.x0, spec.y0) for ln in det.lines
                    ]
                all_items.append(
                    Item(
                        id="tmp",
                        type="text",
                        text=det.text or None,
                        polygon=global_poly,
                        bbox=polygon_to_bbox(global_poly),
                        confidence=det.confidence,
                        source="paddleocr",
                        tile_index=spec.index,
                        granularity=det.granularity,
                        lines=global_lines,
                        recognized=det.recognized,
                        crop_b64=det.crop_b64,
                    )
                )

            # --- qr / barcode (best-effort; missing lib just skips this tile) ---
            if codes is not None:
                try:
                    for det in codes.detect(tile):
                        global_poly = offset_polygon(
                            det.polygon, spec.x0, spec.y0
                        )
                        all_items.append(
                            Item(
                                id="tmp",
                                type=det.type,  # "qr" | "barcode"
                                content=det.content,
                                polygon=global_poly,
                                bbox=polygon_to_bbox(global_poly),
                                confidence=det.confidence,
                                source="pyzbar",
                                tile_index=spec.index,
                            )
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "Code detection failed on tile %d, skipping: %s",
                        spec.index, exc,
                    )
        return all_items, grid

    def _maybe_vlm_fallback(
        self, img, items: list[Item]
    ) -> tuple[list[Item], int, dict]:
        """Re-recognize hard text regions via the VLM: low-confidence crops AND
        circular/ring-shaped regions.

        Two kinds of "hard" regions are collected and sent to the VLM in one
        batched, concurrent pass (8-way thread pool — N crops = N independent
        HTTP calls, NOT a packed multi-image request):

        - **Low-confidence suspects**: text items below ``rec_confidence_fallback``
          (default 0.95). Sent with the provider's built-in art-text prompt.
        - **Circular regions**: rings of text around logos/seals/badges, found by
          :func:`app.regions.detect_circular_regions` (pure geometry). Each whole
          ring is cropped as its bounding box and sent with ``_CIRCULAR_PROMPT``
          — the VLM reads the arc-arranged characters. Members of a detected
          ring are EXCLUDED from the low-confidence set so they aren't sent
          twice.

        Geometry (polygon/bbox) ALWAYS stays from PaddleOCR — the VLM only
        supplies text. Results are written back with ``source="vlm_fallback"``.
        For a circular region, the recognized ring text goes onto ONE
        representative member (the top-most); other members keep their original
        text to avoid duplicating the ring string across several boxes.

        Returns ``(items, n_crops, stats)``: ``n_crops`` is the total number of
        crops sent (suspects + rings), so the timing log reflects how many
        regions were inspected. ``stats`` carries the breakdown
        (sent/rescued/empty/suspects/rings/threshold) for the /logs UI; it's an
        empty dict when fallback didn't run. On any VLM/circle failure the
        originals are kept (best-effort — this stage must never break the main
        OCR path).
        """
        if not self.settings.vlm_enabled or not self.settings.vlm_ocr_fallback_enabled:
            return items, 0, {}
        try:
            vlm = self._get_vlm()
        except Exception as exc:  # noqa: BLE001 — VLM is best-effort
            logger.warning("VLM unavailable, skipping fallback: %s", exc)
            return items, 0, {}

        threshold = self.settings.rec_confidence_fallback

        # --- circular regions: find rings first so their members can be pulled
        # out of the low-confidence suspect set (avoid double-sending). ---
        circular = []
        try:
            from .regions import detect_circular_regions
            circular = detect_circular_regions(img, items, self.settings)
        except Exception as exc:  # noqa: BLE001 — gain-only; never break OCR
            logger.warning("circular detection failed, skipping: %s", exc)
        circle_member_idx: set[int] = set()
        for r in circular:
            circle_member_idx.update(r.member_indices)

        # --- low-confidence suspects (excluding ring members) ---
        suspect_idx = [
            i for i, it in enumerate(items)
            if it.type == "text"
            and it.source == "paddleocr"
            and it.confidence < threshold
            and i not in circle_member_idx
        ]

        # Build one (polygon, prompt) list for both kinds → single batched call.
        # Low-confidence suspects use the default prompt (empty string sentinel
        # → recognize_crop's built-in prompt); circular regions use the arc one.
        from .vlm.qwen import _PROMPT as _DEFAULT_PROMPT
        crops: list[tuple[list[list[float]], str]] = []
        crops.extend((items[i].polygon, _DEFAULT_PROMPT) for i in suspect_idx)
        crops.extend((r.polygon, _CIRCULAR_PROMPT) for r in circular)

        if not crops:
            return items, 0, {}

        try:
            recognized = vlm.recognize_crops_with_prompts_batch(img, crops)
        except Exception as exc:  # noqa: BLE001 — best-effort; keep originals
            logger.warning(
                "vlm fallback FAILED: sent=%d all kept (originals), error: %s",
                len(crops), exc,
            )
            return items, len(crops), {
                "sent": len(crops), "rescued": 0, "empty": 0,
                "suspects": len(suspect_idx), "rings": len(circular),
                "threshold": threshold,
            }

        # --- write results back (geometry unchanged) ---
        out = list(items)
        n_rescued = n_empty = 0
        # Per-crop breakdown for the /logs UI detail view. Bounded in length
        # below; the aggregate counts stay complete regardless.
        crop_details: list[dict] = []
        # Low-confidence suspects: 1:1 text replacement.
        for idx, (new_text, new_conf) in zip(suspect_idx, recognized[:len(suspect_idx)]):
            it = out[idx]
            if new_text:
                n_rescued += 1
                out[idx] = it.model_copy(
                    update={
                        "text": new_text,
                        "confidence": max(it.confidence, new_conf),
                        "source": "vlm_fallback",
                    }
                )
                outcome = "rescued"
            else:
                n_empty += 1
                outcome = "empty"
            crop_details.append({
                "kind": "suspect",
                "box": [int(round(c)) for c in it.bbox],
                "orig_text": it.text,
                "orig_conf": round(float(it.confidence), 3),
                "vlm_text": new_text,
                "vlm_conf": round(float(new_conf), 3),
                "outcome": outcome,
            })
        # Circular regions: the VLM read the WHOLE ring as one string. Put it on
        # the representative member (top-most by bbox y1); leave other members'
        # text alone so the ring string isn't duplicated across boxes.
        circle_results = recognized[len(suspect_idx):]
        for region, (new_text, new_conf) in zip(circular, circle_results):
            # Region bbox = union of member bboxes (for the UI to highlight).
            mb = [items[i].bbox for i in region.member_indices]
            rbox = [min(b[0] for b in mb), min(b[1] for b in mb),
                    max(b[2] for b in mb), max(b[3] for b in mb)] if mb else [0, 0, 0, 0]
            if not new_text or not region.member_indices:
                outcome = "empty"
            else:
                n_rescued += 1
                outcome = "rescued"
                rep = min(region.member_indices, key=lambda i: items[i].bbox[1])
                out[rep] = out[rep].model_copy(
                    update={
                        "text": new_text,
                        "confidence": max(out[rep].confidence, new_conf),
                        "source": "vlm_fallback",
                    }
                )
            crop_details.append({
                "kind": "ring",
                "box": [int(round(c)) for c in rbox],
                "orig_text": "(arc)",
                "orig_conf": None,
                "vlm_text": new_text,
                "vlm_conf": round(float(new_conf), 3),
                "outcome": outcome,
                "members": len(region.member_indices),
            })
        logger.info(
            "vlm fallback: sent=%d rescued=%d empty=%d "
            "(suspects=%d rings=%d threshold=%.2f)",
            len(crops), n_rescued, n_empty,
            len(suspect_idx), len(circular), threshold,
        )
        from .log_buffer import CAPACITY_CROPS
        return out, len(crops), {
            "sent": len(crops),
            "rescued": n_rescued,
            "empty": n_empty,
            "suspects": len(suspect_idx),
            "rings": len(circular),
            "threshold": threshold,
            "crops": crop_details[:CAPACITY_CROPS],
        }

    def _drop_low_confidence(self, items: list[Item]) -> list[Item]:
        """Discard text items whose final confidence is below rec_confidence_drop.

        Called only on the POST /analyze path (``confidence_policy=True``). Runs
        AFTER the VLM fallback pass, so a box the VLM rescued above the drop
        threshold survives; one that stayed low (either never sent or VLM-empty)
        is removed. Only ``type == "text"`` items are dropped — qr/barcode
        confidence has different semantics and those decoded payloads are
        valuable regardless of score.

        The drop threshold is clamped to ``rec_confidence_fallback`` so a
        misconfiguration (drop > fallback) can't silently widen the re-read set.
        """
        drop = min(
            self.settings.rec_confidence_drop,
            self.settings.rec_confidence_fallback,
        )
        kept = [
            it for it in items
            if it.type != "text" or it.confidence >= drop
        ]
        n_drop = len(items) - len(kept)
        if n_drop:
            logger.info(
                "confidence policy: dropped %d/%d text items "
                "(confidence < %.2f; kept qr/barcode)",
                n_drop, len(items), drop,
            )
        return kept


def _b64_png(pil_img) -> str:
    import io
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")
