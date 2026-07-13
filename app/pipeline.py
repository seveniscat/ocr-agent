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
    ) -> AnalyzeResponse:
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
        all_items, vlm_calls = self._maybe_vlm_fallback(img, all_items)
        t_vlm = time.perf_counter() - t_vlm

        # --- final dedupe (paragraph blocks can still overlap at tile seams) ---
        all_items = dedupe_items(
            all_items,
            merge_x_thres=merge_x,
            merge_y_thres=merge_y,
        )
        all_items = renumber(all_items, prefix="t")
        t_dedup = time.perf_counter() - t_dedup

        # --- per-stage timing log. vlm_calls is the number of CROPS inspected;
        # they're sent in ~⌈crops/batch⌉ batched requests (each ~20 crops in
        # one multi-image call), so wall-clock no longer scales linearly with it. ---
        t_total = time.perf_counter() - t0
        logger.info(
            "pipeline.run: %dx%d→%dx%d tiles=%d items=%d→%d "
            "preprocess=%.2fs ocr=%.2fs vlm(crops=%d)=%.2fs dedupe=%.2fs total=%.2fs",
            orig_w, orig_h, w, h, grid.count, n_after_ocr, len(all_items),
            t_pre, t_ocr, vlm_calls, t_vlm, t_dedup, t_total,
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
    ) -> tuple[list[Item], int]:
        """Re-recognize low-confidence text crops via the VLM, in BATCH.

        Collects every text item below the confidence threshold, sends ALL of
        them to the VLM in as few multi-image calls as possible
        (``recognize_crops_batch`` packs ~20 crops per request), then writes the
        recognized text back into the items. Geometry (polygon/bbox) stays from
        the expert detector — the VLM only supplies text.

        Items stay ``type=text`` (not ``art_text``): art_text is for stylized
        glyphs; VLM here is a confidence booster (e.g. wrong Paddle lang model).

        Returns ``(items, n_crops)``: ``n_crops`` is the number of crops sent
        (not the number of API calls), so the timing log still shows how many
        suspicious regions were inspected. The actual speedup is in the API
        call count, which is ~⌈n_crops / batch_size⌉ instead of n_crops.
        """
        if not self.settings.vlm_enabled or not self.settings.vlm_ocr_fallback_enabled:
            return items, 0
        try:
            vlm = self._get_vlm()
        except Exception as exc:  # noqa: BLE001 — VLM is best-effort
            logger.warning("VLM unavailable, skipping fallback: %s", exc)
            return items, 0

        threshold = self.settings.rec_confidence_fallback
        # Collect the indices + polygons of every suspect item, then send them
        # to the VLM in one batched call (the provider chunks internally).
        suspect_idx = [
            i for i, it in enumerate(items)
            if it.type == "text"
            and it.source == "paddleocr"
            and it.confidence < threshold
        ]
        if not suspect_idx:
            return items, 0

        suspect_polys = [items[i].polygon for i in suspect_idx]
        try:
            recognized = vlm.recognize_crops_batch(img, suspect_polys)
        except Exception as exc:  # noqa: BLE001 — best-effort; keep originals
            logger.warning("VLM batch fallback failed, keeping originals: %s", exc)
            return items, len(suspect_idx)

        # Write the recognized text back into the items (geometry unchanged).
        out = list(items)
        for idx, (new_text, new_conf) in zip(suspect_idx, recognized):
            if new_text:
                out[idx] = out[idx].model_copy(
                    update={
                        "text": new_text,
                        "confidence": max(out[idx].confidence, new_conf),
                        "source": "vlm_fallback",
                    }
                )
        return out, len(suspect_idx)


def _b64_png(pil_img) -> str:
    import io
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")
