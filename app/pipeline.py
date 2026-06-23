"""Pipeline orchestration.

Flow:
    load image → plan grid → for each tile:
        detect text (polygon) → recognize → collect items
        detect qr/barcode → collect items
    (after tiles) low-confidence text → VLM fallback
    → map to global coords → dedupe (NMS) → annotate (optional) → return

The pipeline is a thin coordinator: each stage lives in its own module so it
can be swapped or unit-tested independently.
"""
from __future__ import annotations

import base64
import logging

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
    ) -> AnalyzeResponse:
        img = load_image(image_data)
        h, w = img.shape[:2]

        # --- Preprocess: crop blank margins around the die-line artwork. ---
        # The cropped image becomes the working image for everything below, so
        # all item coordinates are naturally in cropped space; the one `crop`
        # offset is echoed back in image_meta for callers to remap to the
        # original. Blank image → no crop, fall through unchanged.
        crop_box = None
        if self.settings.preprocess_autocrop:
            from .preprocess import autocrop

            img, crop_box = autocrop(
                img,
                threshold=self.settings.preprocess_autocrop_threshold,
                padding=self.settings.preprocess_autocrop_padding,
            )
            h, w = img.shape[:2]

        grid = plan_grid(
            w, h,
            target_size=self.settings.tile_target_size,
            overlap=self.settings.tile_overlap,
        )
        specs = tile_specs(grid)

        all_items: list[Item] = []

        ocr = self._get_ocr()
        codes = self._get_codes_or_none()  # may be None (lib missing) → skip

        # Translate per-request OCR overrides to detector call args.
        if options is not None:
            predict_kwargs = options.to_predict_kwargs()
            gran = options.granularity
            gap = options.paragraph_gap_ratio
            xov = options.paragraph_x_overlap
        else:
            predict_kwargs, gran, gap, xov = {}, None, None, None

        # Per-tile processing.
        for spec in specs:
            tile = crop_tile(img, spec)

            # --- text (primary channel; errors here are fatal) ---
            for det in ocr.detect_and_recognize(
                tile,
                predict_kwargs=predict_kwargs,
                granularity=gran,
                paragraph_gap_ratio=gap,
                paragraph_x_overlap=xov,
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
                        text=det.text,
                        polygon=global_poly,
                        bbox=polygon_to_bbox(global_poly),
                        confidence=det.confidence,
                        source="paddleocr",
                        tile_index=spec.index,
                        granularity=det.granularity,
                        lines=global_lines,
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

        # --- VLM fallback for low-confidence / suspicious text ---
        all_items = self._maybe_vlm_fallback(img, all_items)

        # --- merge duplicates across tile seams ---
        all_items = dedupe_items(all_items)
        all_items = renumber(all_items, prefix="t")

        response = AnalyzeResponse(
            image_meta=ImageMeta(
                width=w, height=h, tile_count=grid.count, crop=crop_box
            ),
            items=all_items,
            options_used=options,
        )

        if annotate:
            from .viz.annotator import annotate_image

            annotated = annotate_image(img, all_items, self.settings)
            response.annotated_image_b64 = _b64_png(annotated)

        return response

    # -- helpers ------------------------------------------------------------

    def _maybe_vlm_fallback(
        self, img, items: list[Item]
    ) -> list[Item]:
        """For text items below the confidence threshold, re-recognize the
        crop via the VLM. Keeps the original polygon (geometry stays from the
        expert detector — VLM only supplies the text)."""
        if not self.settings.vlm_enabled:
            return items
        try:
            vlm = self._get_vlm()
        except Exception as exc:  # noqa: BLE001 — VLM is best-effort
            logger.warning("VLM unavailable, skipping fallback: %s", exc)
            return items

        threshold = self.settings.rec_confidence_fallback
        out: list[Item] = []
        for item in items:
            if (
                item.type == "text"
                and item.source == "paddleocr"
                and item.confidence < threshold
            ):
                try:
                    new_text, new_conf = vlm.recognize_crop(img, item.polygon)
                    if new_text:
                        item = item.model_copy(
                            update={
                                "text": new_text,
                                "confidence": max(item.confidence, new_conf),
                                "type": "art_text",
                                "source": "vlm_fallback",
                            }
                        )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("VLM fallback failed for one item: %s", exc)
            out.append(item)
        return out


def _b64_png(pil_img) -> str:
    import io
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")
