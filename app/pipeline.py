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
import re
import time
import unicodedata

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


# Strips leading/trailing whitespace + punctuation (ASCII and Unicode) while
# leaving letters/digits of ALL scripts (incl. CJK, Hangul, Hiragana) intact.
#
# We CANNOT use ``\\W`` for this: under the default Unicode str semantics,
# ``\\W`` matches every non-ASCII character, so a single CJK char like "京"
# would be stripped — silently dropping valid single-character labels
# (license-plate prefixes, seals, single-char copy) as "junk". Instead we
# build an explicit class from Unicode general categories: whitespace, all
# Punctuation (P*), and all Symbol (S* — covers currency/math symbols that
# detectors emit as standalone noise). Built once at import time.
def _build_edge_junk_pattern() -> "re.Pattern[str]":
    chars = []
    for cp in range(0x110000):
        ch = chr(cp)
        cat = unicodedata.category(ch)
        # Zs/Tab/NBSP whitespace
        if cat == "Zs" or ch in "\t\n\r\f\v ":
            chars.append(ch)
        # Punctuation (Pd/Ps/Pe/Pi/Pf/Pc/Po) and Symbols (Sm/Sc/Sk/So)
        elif cat[0] in ("P", "S"):
            chars.append(ch)
    # Escape + build; the set is large but compiled once.
    return re.compile(
        "^[" + re.escape("".join(chars)) + "]+|"
        "[" + re.escape("".join(chars)) + "]+$"
    )


_EDGE_JUNK_RE = _build_edge_junk_pattern()

# Prompt for the circular-region VLM read. A whole ring (logo/seal/badge text on
# an arc) is cropped as its bounding box and sent with this prompt — the VLM is
# asked to read the characters around the ring. We do NOT polar-unroll here
# (path A): rely on the VLM's understanding of arc layout. If this proves
# unreliable on dense rings, the upgrade path is cv2.warpPolar before sending.
_CIRCULAR_PROMPT = (
    "这是包装上沿圆弧/环形排布的文字（圆形 logo、印章、徽章上的弧形文字）。"
    "请按顺时针方向，从顶部（12 点钟方向）开始，原样读出环上所有可见文字。"
    "用 / 分隔各段弧形文字（如顶部和底部是两段）。"
    "只有当你能确信读对每一个字时才返回结果；只要有一个字模糊、被遮挡、"
    "或需要靠猜，就不要返回部分或臆测的内容，直接输出: EMPTY"
    "（准确的空，优于错误的猜）。"
    "返回文字时只输出文字本身，不要解释、不要引号、不要 JSON、不要代码围栏。"
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
        for_verify: bool = False,
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

            cropped, cbox = autocrop(
                img,
                threshold=self.settings.preprocess_autocrop_threshold,
                padding=self.settings.preprocess_autocrop_padding,
            )
            ch, cw = cropped.shape[:2]
            # Guard against pathological autocrop results (near-blank images that
            # crop down to a few noise pixels). Below this minimum the cropped
            # canvas is too small for meaningful detection; fall back to the
            # pre-crop image so downstream never sees a degenerate tiny frame.
            _AUTOCROP_MIN_SIDE = 32
            if min(cw, ch) >= _AUTOCROP_MIN_SIDE:
                img, crop_box = cropped, cbox
                h, w = ch, cw
            else:
                logger.warning(
                    "autocrop produced %dx%d (below min %d); using original %dx%d",
                    cw, ch, _AUTOCROP_MIN_SIDE, w, h,
                )
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
        all_items, vlm_calls, vlm_stats = self._maybe_vlm_fallback(
            img, all_items, for_verify=for_verify
        )
        t_vlm = time.perf_counter() - t_vlm

        # --- final dedupe (paragraph blocks can still overlap at tile seams) ---
        all_items = dedupe_items(
            all_items,
            merge_x_thres=merge_x,
            merge_y_thres=merge_y,
        )

        # --- universal content-quality cleanup (ALL paths). Runs before the
        # confidence policy so the policy operates on already-clean items. Drops
        # empty-text, unrecognized, junk-short, and low-confidence text items so
        # every caller — sync /analyze, async /analyze, even /verify — gets clean
        # results. /verify opts out of the short-text/low-confidence cuts
        # (keep_short=True) because it needs every recognizable char to match
        # required copy, but empty/unrecognized boxes still go (they contribute
        # zero chars to matching). qr/barcode are never touched here. ---
        n_before_clean = len(all_items)
        all_items, clean_breakdown = self._clean_items(
            all_items, keep_short=for_verify, options=options
        )
        n_cleaned = n_before_clean - len(all_items)

        # --- confidence policy (POST /analyze only): the SINGLE /analyze-specific
        # cut. The universal floor (min_keep_confidence) already ran in
        # _clean_items above, so every remaining text box is at or above that
        # floor; re-applying rec_confidence_drop here would be redundant when the
        # two defaults match (0.6). What /analyze adds on top is the VLM-gate:
        # a box the VLM looked at but couldn't read (vlm_lifted is False) AND
        # whose confidence is still below rec_confidence_vlm_drop — nobody could
        # read it confidently, so /analyze discards it. /verify opts out
        # (confidence_policy=False) because it needs every OCR'd character.
        # qr/barcode are never dropped (different confidence semantics). ---
        n_dropped = 0
        vlm_drop_threshold_used = 0.0
        if confidence_policy:
            n_before_drop = len(all_items)
            all_items = self._drop_low_confidence(all_items)
            n_dropped = n_before_drop - len(all_items)
            vlm_drop_threshold_used = self.settings.rec_confidence_vlm_drop

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
            "vlm(crops=%d)=%.2fs dedupe=%.2fs clean=%d drop=%d total=%.2fs",
            orig_w, orig_h, w, h, grid.count, n_after_ocr, len(all_items),
            t_pre, t_ocr, _bd, _br, vlm_calls, t_vlm, t_dedup, n_cleaned,
            n_dropped, t_total,
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
                "vlm_drop_threshold": vlm_drop_threshold_used,
                "clean_empty": clean_breakdown.get("empty", 0),
                "clean_unrec": clean_breakdown.get("unrec", 0),
                "clean_short": clean_breakdown.get("short", 0),
                "clean_small": clean_breakdown.get("small", 0),
                "clean_low": clean_breakdown.get("low", 0),
            })

        # Echo the funnel diagnostics back to the caller (built just above).
        # The WebUI debug view renders this; API callers ignore it. Attached
        # AFTER stats_sink is populated so the snapshot is complete.
        if stats_sink is not None:
            response.stats = dict(stats_sink)

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
        self, img, items: list[Item], *, for_verify: bool = False
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
        # Floor below which sending a crop to the VLM is pure waste: a box the
        # universal cleanup will discard anyway (confidence < min_keep_confidence)
        # goes no matter what the VLM returns — the VLM no longer contributes to
        # confidence (it's the pure PaddleOCR score), so a 0.40 box stays 0.40
        # and can't clear the universal floor. Skip the cloud call.
        #
        # NOTE: this floor is intentionally ONLY the universal min_keep_confidence,
        # NOT the /analyze-specific rec_confidence_drop. The /analyze drop policy
        # is applied AFTER the VLM pass (see _drop_low_confidence), so a box in
        # [min_keep_confidence, rec_confidence_drop) deserves a VLM second
        # opinion before being dropped — seals/art text are systematically
        # low-scored by PaddleOCR yet readable by the VLM. Folding rec_confidence_
        # drop into the floor would starve that band of VLM rescues.
        #
        # /verify is EXEMPT from this floor: it runs cleanup with keep_short=True
        # (low-confidence text is kept) because it needs every readable char to
        # match required copy, so a low PaddleOCR box that the VLM CAN read is
        # valuable there.
        if for_verify:
            vlm_floor = 0.0
        else:
            vlm_floor = self.settings.min_keep_confidence

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
        # Only re-read boxes in [vlm_floor, threshold): below vlm_floor they'd
        # be discarded anyway (pure waste of a cloud call); above threshold
        # PaddleOCR is confident enough. Ring members are handled separately.
        # Also skip boxes that will be dropped by the universal small-box filter
        # (min_box_side) — sending a 3px speck to the VLM is pure waste too.
        min_side = self.settings.min_box_side if not for_verify else 0
        def _too_small(it):
            if min_side <= 0:
                return False
            bx0, by0, bx1, by1 = it.bbox
            return (bx1 - bx0) < min_side and (by1 - by0) < min_side
        suspect_idx = [
            i for i, it in enumerate(items)
            if it.type == "text"
            and it.source == "paddleocr"
            and vlm_floor <= it.confidence < threshold
            and not _too_small(it)
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
        # The VLM does text recognition only — it returns no usable score, so
        # "success" = "non-empty text". Confidence is NEVER taken from the VLM:
        # we keep the original PaddleOCR score for the box regardless. vlm_lifted
        # thus means "the VLM produced a read" (True), not "it beat paddleocr".
        for idx, (new_text, _new_conf) in zip(suspect_idx, recognized[:len(suspect_idx)]):
            it = out[idx]
            if new_text:
                n_rescued += 1
                out[idx] = it.model_copy(
                    update={
                        "text": new_text,
                        # Confidence stays the original PaddleOCR score — the
                        # VLM supplies text, not a trustworthy confidence.
                        "source": "vlm_fallback",
                        "vlm_lifted": True,
                    }
                )
                outcome = "rescued"
            else:
                n_empty += 1
                # VLM returned empty (or its output was rejected as garbage):
                # flag vlm_lifted=False so the /analyze rule-2 drop policy can
                # catch it. Don't touch text/confidence/source — keep the
                # original PaddleOCR values.
                out[idx] = it.model_copy(update={"vlm_lifted": False})
                outcome = "empty"
            crop_details.append({
                "kind": "suspect",
                "box": [int(round(c)) for c in it.bbox],
                "orig_text": it.text,
                "orig_conf": round(float(it.confidence), 3),
                "vlm_text": new_text,
                "outcome": outcome,
            })
        # Circular regions: the VLM read the WHOLE ring as one string. Put it on
        # the representative member (top-most by bbox y1); leave other members'
        # text alone so the ring string isn't duplicated across boxes.
        circle_results = recognized[len(suspect_idx):]
        for region, (new_text, _new_conf) in zip(circular, circle_results):
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
                        # Confidence stays the original PaddleOCR score — the
                        # VLM supplies text, not a trustworthy confidence.
                        "source": "vlm_fallback",
                    }
                )
            crop_details.append({
                "kind": "ring",
                "box": [int(round(c)) for c in rbox],
                "orig_text": "(arc)",
                "orig_conf": None,
                "vlm_text": new_text,
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

    def _clean_items(
        self, items: list[Item], *, keep_short: bool = False,
        options: "OCROptions | None" = None,
    ) -> tuple[list[Item], dict]:
        """Universal content-quality cleanup. Runs on ALL paths.

        Drops text/art_text items that are empty, unrecognized, junk-short, or
        tiny boxes (detector noise), so every caller receives clean results
        without filtering themselves. qr/barcode are NEVER dropped here (decoded
        payloads are valuable regardless of score).

        Rules (only ``type in (text, art_text)`` is considered):

        1. Empty text — ``text`` is None or whitespace-only.
        2. Unrecognized — ``recognized is False`` (detector boxed it but the
           recognizer dropped it; text is empty, confidence is 0).
        3. Junk-short (skipped when ``keep_short=True``) — after stripping
           leading/trailing whitespace AND punctuation, fewer than
           ``min_text_chars`` effective characters remain (e.g. '.', '，', '-',
           a lone ASCII digit). Catches detector noise.
        4. Small box (skipped when ``keep_short=True``) — bbox with EITHER
           dimension (width OR height) shorter than ``min_box_side`` pixels.
           Catches detector crumbs/specks on textured backgrounds. Narrow-but-
           long boxes (a column of chars, a thin line) are NOT dropped.

        There is NO universal confidence floor anymore: low-confidence but
        readable text (seals, art text, blurry-but-decodable chars) is kept and
        left to the per-path VLM-gate policy (``_drop_low_confidence``) and the
        VLM fallback re-read to decide. ``min_keep_confidence`` remains as a
        knob but defaults to 0.0 (disabled).

        ``keep_short=True`` is used by /verify, which needs every recognizable
        character to match required copy (rules 3-4 exempt), but empty and
        unrecognized boxes still go — they contribute zero chars to matching.

        ``options`` carries per-call overrides for rules 3-4
        (``min_text_chars`` / ``min_box_side`` / ``min_keep_confidence``); a
        None field falls back to the Settings (.env) default, so omitting them
        is backward compatible.

        Returns ``(kept, breakdown)`` where ``breakdown`` carries the per-rule
        drop counts (``empty`` / ``unrec`` / ``short`` / ``small`` / ``low``)
        for stats_sink observability.
        """
        # Per-call override → .env default. None on the option means "don't
        # override" (use the server default), which keeps old callers working.
        min_chars = (
            options.min_text_chars
            if options and options.min_text_chars is not None
            else self.settings.min_text_chars
        )
        min_side = (
            options.min_box_side
            if options and options.min_box_side is not None
            else self.settings.min_box_side
        )
        min_conf = (
            options.min_keep_confidence
            if options and options.min_keep_confidence is not None
            else self.settings.min_keep_confidence
        )
        # Leading/trailing whitespace + punctuation to strip when measuring the
        # "effective" character count for the junk-short rule.
        #
        # IMPORTANT: do NOT use ``\\W`` here. Under ``re.UNICODE`` (the default
        # for str patterns in Py3), ``\\W`` matches EVERY non-ASCII character —
        # so a lone CJK character like "京" or "沪" (license-plate prefix, seal
        # text, single-char labels) is treated as "non-word" and stripped to an
        # empty string, then wrongly dropped as junk. Instead we strip an
        # explicit punctuation class: ASCII punctuation + the Unicode general
        # categories P* (Punctuation) and S* (Symbol), leaving letters/digits
        # of ALL scripts intact. ``regex`` supports \\p{...}; the stdlib ``re``
        # does not, so we build the class from unicodedata at import time.
        _edge_junk = _EDGE_JUNK_RE

        kept: list[Item] = []
        n_empty = n_unrec = n_short = n_small = n_low = 0
        for it in items:
            # qr/barcode: always keep, regardless of content or confidence.
            if it.type not in ("text", "art_text"):
                kept.append(it)
                continue
            # Rule 1: empty text.
            if not it.text or not it.text.strip():
                n_empty += 1
                continue
            # Rule 2: unrecognized box.
            if not it.recognized:
                n_unrec += 1
                continue
            if not keep_short:
                # Rule 4 (checked before junk-short, since a tiny box is noise
                # regardless of its text content): BOTH bbox dimensions below
                # the floor → drop. "min side" on purpose: a box tiny in only
                # one axis (narrow-but-tall column, short-but-wide line) is
                # legitimate copy and is NOT dropped; only crumbs/specks small
                # in BOTH axes are. min_box_side=0 disables this.
                if min_side > 0:
                    bx0, by0, bx1, by1 = it.bbox
                    bw, bh = bx1 - bx0, by1 - by0
                    if bw < min_side and bh < min_side:
                        n_small += 1
                        continue
                # Rule 3: junk-short text (noise like lone punctuation/digits).
                if min_chars > 0:
                    effective = _edge_junk.sub("", it.text)
                    if len(effective) < min_chars:
                        n_short += 1
                        continue
                # Optional confidence floor (default disabled). Kept as an
                # escape hatch for callers that want a hard score gate.
                if min_conf > 0.0 and it.confidence < min_conf:
                    n_low += 1
                    continue
            kept.append(it)

        n_total = len(items) - len(kept)
        if n_total:
            rules = (
                f"empty={n_empty} unrec={n_unrec}"
                + ("" if keep_short else f" short={n_short} small={n_small} low={n_low}")
            )
            mode = "verify(keep_short)" if keep_short else "full"
            logger.info(
                "clean_items[%s]: dropped %d/%d text items (%s); qr/barcode kept",
                mode, n_total, len(items), rules,
            )
        return kept, {
            "empty": n_empty, "unrec": n_unrec,
            "short": n_short, "small": n_small, "low": n_low,
        }

    def _drop_low_confidence(self, items: list[Item]) -> list[Item]:
        """Discard text items per the /analyze confidence policy.

        Called only on the POST /analyze path (``confidence_policy=True``).
        Runs AFTER the VLM fallback pass AND AFTER the universal
        ``_clean_items`` floor (``min_keep_confidence``), so every remaining
        text item is already at or above that universal floor.

        The SINGLE /analyze-specific rule (the universal floor does NOT cover
        it): a box that was SENT to the VLM fallback but produced no readable
        text (``vlm_lifted is False``: the VLM returned empty/garbage), AND
        whose confidence is still below ``rec_confidence_vlm_drop`` (default
        0.85). Since the VLM no longer contributes to ``confidence`` (it's the
        pure PaddleOCR score), this drops boxes where BOTH PaddleOCR was unsure
        AND the VLM couldn't read it either — nobody could read it confidently.

        Why ``rec_confidence_drop`` is no longer applied here: the universal
        ``_clean_items`` rule-4 already cuts everything below
        ``min_keep_confidence`` on ALL paths (including /analyze). When the two
        defaults match (0.6), a second cut here was pure redundancy; when they
        differ, the universal rule is the correct single source of truth for
        "never keep below this". /analyze's ONLY addition is the VLM-gate above.

        Only ``type == "text"`` items are dropped — ``art_text`` is treated as
        high-value copy (it survived the detector's art-text path) and qr/barcode
        confidence has different semantics; those decoded payloads are valuable
        regardless of score.
        """
        vlm_drop = self.settings.rec_confidence_vlm_drop
        kept = [
            it for it in items
            if it.type != "text"
            # The single rule: a box the VLM looked at but couldn't read
            # (``vlm_lifted is False``) AND the PaddleOCR score is also below
            # vlm_drop — nobody could read it confidently. Drop it.
            or not (it.vlm_lifted is False and it.confidence < vlm_drop)
        ]
        n_drop = len(items) - len(kept)
        if n_drop:
            logger.info(
                "confidence policy: dropped %d/%d text items "
                "(vlm-not-lifted & conf<%.2f; kept qr/barcode/art_text)",
                n_drop, len(items), vlm_drop,
            )
        return kept


def _b64_png(pil_img) -> str:
    import io
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")
