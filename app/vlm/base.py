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
