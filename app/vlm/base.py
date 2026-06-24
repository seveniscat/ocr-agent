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
