"""VLM provider abstraction.

Only used as a **fallback** for art / curved text that PaddleOCR can't read
confidently. The provider takes a high-res crop of the suspect polygon and
returns recognized text + a confidence. The polygon itself stays from the
expert detector (we never trust VLM-emit coordinates — see project plan).

v1 ships an OpenAI-compatible implementation (works with Qwen-VL via
DashScope, and many others by swapping base_url). Add new providers by
subclassing :class:`VLMProvider` and adding a branch in :func:`build_vlm`.
"""
from __future__ import annotations

import abc
from typing import TYPE_CHECKING

from ..config import Settings

if TYPE_CHECKING:
    import numpy as np


class VLMProvider(abc.ABC):
    """Recognize text in a cropped region. Returns (text, confidence)."""

    name: str = "base"

    # Minimum width AND height (px) of a crop we'll send to the VLM. 0 = no
    # check. Subclasses (QwenVLM) override from settings.vlm_min_crop_side so
    # the threshold is configurable. Crops below this are silently dropped
    # (returned as empty) — see _crop_bbox_or_none for the why.
    _min_crop_side: int = 0

    def _crop_bbox_or_none(
        self, image: "np.ndarray", polygon: list[list[float]]
    ) -> tuple[int, int, int, int] | None:
        """Axis-aligned bbox of ``polygon`` clamped to ``image``, or None.

        Returns None when the bbox is empty/degenerate OR smaller than
        ``self._min_crop_side`` on either axis. The min-side guard exists
        because DashScope (and most VLM endpoints) reject images below ~10px
        with HTTP 400, and the batch wrapper treats one failure as total
        failure — so a single 7x36 crop poisons the whole fallback batch.
        Crops that small can't hold legible characters anyway, so dropping
        them loses nothing.
        """
        xs = [p[0] for p in polygon]
        ys = [p[1] for p in polygon]
        x1, y1, x2, y2 = int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))
        h, w = image.shape[:2]
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        if self._min_crop_side > 0 and (
            x2 - x1 < self._min_crop_side or y2 - y1 < self._min_crop_side
        ):
            return None
        return x1, y1, x2, y2

    @abc.abstractmethod
    def recognize_crop(
        self, image: "np.ndarray", polygon: list[list[float]]
    ) -> tuple[str, float]:
        """Read text inside ``polygon`` (in ``image`` coords)."""
        raise NotImplementedError

    def recognize_crops_batch(
        self, image: "np.ndarray", polygons: list[list[list[float]]]
    ) -> list[tuple[str, float]]:
        """Recognize text in multiple crops concurrently.

        Default implementation dispatches ``recognize_crop`` calls across a
        thread pool so N independent cloud round-trips overlap instead of
        waiting on each other. Returns one ``(text, confidence)`` per input
        polygon, in order. Providers MAY override with a true multi-image
        single-request path — but for VLM art-text fallback, concurrency wins
        in practice (a multi-image request forces serial cross-image attention
        server-side and is often slower than parallel single-image calls).
        """
        import concurrent.futures as cf

        if not polygons:
            return []
        # Each call is I/O-bound (waiting on the cloud API), so a thread pool
        # is the right tool. 8 workers balances throughput against rate limits.
        with cf.ThreadPoolExecutor(max_workers=8) as pool:
            futures = [
                pool.submit(self.recognize_crop, image, poly) for poly in polygons
            ]
            return [f.result() for f in futures]

    def recognize_crop_with_prompt(
        self, image: "np.ndarray", polygon: list[list[float]], prompt: str
    ) -> tuple[str, float]:
        """Read text in a crop using a CALLER-SUPPLIED prompt.

        Unlike :meth:`recognize_crop` (which uses the provider's built-in
        art-text prompt), this lets the caller pass a region-specific prompt —
        e.g. the circular-text prompt that tells the VLM the characters are
        arranged on an arc. Crops the axis-aligned bbox of ``polygon`` (same
        crop as ``recognize_crop``), encodes JPEG, and calls ``ask_image``.

        Default implementation works for any provider that implements
        ``ask_image`` (QwenVLM does). Providers without ``ask_image`` raise
        NotImplementedError from ``ask_image``, surfaced to the caller.
        """
        bbox = self._crop_bbox_or_none(image, polygon)
        if bbox is None:
            return "", 0.0
        x1, y1, x2, y2 = bbox

        from .qwen import _to_b64_jpeg  # local import avoids a hard PIL dep at import

        data_url = _to_b64_jpeg(image[y1:y2, x1:x2])
        text, _conf = self.ask_image(
            data_url, prompt, max_tokens=512, json_mode=False
        )
        text = (text or "").strip()
        if not text or text.upper() == "EMPTY":
            return "", 0.0
        # Strip a leading/trailing quote the model sometimes adds.
        import re
        text = re.sub(r"^['\"]|['\"]$", "", text)
        return text, 0.8

    def recognize_crops_with_prompts_batch(
        self,
        image: "np.ndarray",
        crops: list[tuple[list[list[float]], str]],
    ) -> list[tuple[str, float]]:
        """Concurrent version of :meth:`recognize_crop_with_prompt`.

        ``crops`` is a list of ``(polygon, prompt)`` pairs. Mirrors
        :meth:`recognize_crops_batch`: 8-way concurrent dispatch, one result per
        input, in order. Used by the pipeline to send low-confidence suspects
        (default prompt) and circular regions (circular prompt) in one batch.
        """
        import concurrent.futures as cf

        if not crops:
            return []
        with cf.ThreadPoolExecutor(max_workers=8) as pool:
            futures = [
                pool.submit(self.recognize_crop_with_prompt, image, poly, prompt)
                for poly, prompt in crops
            ]
            return [f.result() for f in futures]

    def ask_image(
        self,
        image_b64_data_url: str,
        prompt: str,
        *,
        max_tokens: int = 1024,
        json_mode: bool = False,
        enable_thinking: bool | None = None,
    ) -> tuple[str, float]:
        """Ask the VLM one free-form question about one image.

        Returns ``(raw_text, confidence)``. ``image_b64_data_url`` is a full
        ``data:image/...;base64,...`` URL the provider can drop straight into
        the OpenAI-compatible ``image_url`` slot. ``json_mode`` requests a JSON
        object response where supported. ``enable_thinking`` (Qwen3.x) requests
        deep reasoning before the answer; when both are set, providers should
        honor thinking and drop json_mode (they're mutually exclusive on
        DashScope), relying on the caller's tolerant parser.

        Default raises ``NotImplementedError`` — providers opt in. The
        understanding layer depends on this; ``recognize_crop`` does not, so
        existing providers keep working until they implement it.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not implement ask_image(); "
            "the understanding layer requires an ask_image-capable provider."
        )


def build_vlm(settings: Settings) -> VLMProvider:
    """Factory: pick a provider from ``settings.vlm_provider``."""
    provider = (settings.vlm_provider or "").lower()
    if provider == "qwen":
        from .qwen import QwenVLM

        return QwenVLM(settings)
    raise ValueError(
        f"Unknown VLM provider: {settings.vlm_provider!r}. "
        f"Set OCR_VLM_PROVIDER to one of: qwen"
    )
