"""Image enhancement for OCR preprocessing.

User-selectable enhancement steps that make text crisper / backgrounds calmer
before the OCR detector runs. Each step is a pure function on an RGB uint8
``np.ndarray``; :func:`enhance` chains the enabled ones in a fixed, sensible
order so results stay reproducible.

Implemented with OpenCV + NumPy only (already required by the OCR path — no
new dependency). Default-on steps are the two that help most reliably without
risk of degrading normal text: **CLAHE** (local contrast) and **light
sharpening**. Denoise / grayscale / threshold / morph-close are opt-in because
they trade detail for cleanliness depending on the source.

Design notes:
  - CLAHE runs on the L channel of LAB so contrast is boosted without color
    shift (CLAHE on a packed RGB image amplifies chroma noise unevenly).
  - Sharpen is an unsharp mask (``out = img + amount*(img - blur)``); mild by
    default so hairline strokes don't halo.
  - Denoise uses ``fastNlMeansDenoisingColored`` — edge-preserving, so it
    flattens background speckle without smearing glyph edges.
  - Threshold returns a binary 3-channel image (so the pipeline still sees an
    RGB array); Otsu for uniform lighting, adaptive for uneven.
"""
from __future__ import annotations

import cv2
import numpy as np

# Fixed enhancement order. Grayscale must precede CLAHE/threshold when both
# are on (CLAHE on L is color-aware; once we've collapsed to gray we work on a
# single channel). Sharpen after denoise so we don't sharpen the noise we just
# removed. Threshold near the end (binary output), morph-close last (bridges
# fragmented strokes in the now-binary glyphs).
_DEFAULTS: dict = {
    "clahe": True,
    "clahe_clip": 2.0,
    "clahe_tile": 8,
    "sharpen": True,
    "sharpen_amount": 0.6,
    "sharpen_sigma": 1.0,
    "denoise": False,
    "denoise_strength": 7,
    "grayscale": False,
    "threshold": None,  # None | "otsu" | "adaptive"
    "morph_close": False,
    "morph_ksize": 2,
}


# ---------------------------------------------------------------------------
# helpers
def _to_bgr(img: np.ndarray) -> np.ndarray:
    """RGB→BGR for OpenCV; pass-through for single-channel arrays."""
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def _from_bgr(bgr: np.ndarray) -> np.ndarray:
    """BGR→RGB for our internal convention; pass-through for single channel."""
    if bgr.ndim == 2:
        return bgr
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# steps
def apply_grayscale_to_bgr(img: np.ndarray) -> np.ndarray:
    """Collapse to gray then expand back to 3 channels (keeps the RGB shape).

    A grayscale image as the sole input lets CLAHE/threshold operate on pure
    luminance, and downstream code never has to special-case 1-channel arrays.
    """
    if img.ndim == 2:
        gray = img
    else:
        gray = cv2.cvtColor(_to_bgr(img), cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)


def apply_clahe(
    img: np.ndarray, *, clip: float = 2.0, tile: int = 8
) -> np.ndarray:
    """Contrast-Limited Adaptive Histogram Equalization on the L channel.

    ``tile`` is the per-tile grid size (pixels); larger tiles → broader, more
    global equalization. ``clip`` caps the contrast gain so background noise
    isn't amplified (the "CL" in CLAHE).
    """
    if img.ndim == 2:
        # Single channel: CLAHE directly (still returns 2-D).
        clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(tile, tile))
        return clahe.apply(img)

    lab = cv2.cvtColor(_to_bgr(img), cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip), tileGridSize=(int(tile), int(tile)))
    l = clahe.apply(l)
    merged = cv2.merge((l, a, b))
    return _from_bgr(cv2.cvtColor(merged, cv2.COLOR_LAB2BGR))


def apply_denoise(img: np.ndarray, *, strength: int = 7) -> np.ndarray:
    """Edge-preserving denoise. ``strength`` (h) scales the luminance component.

    ``h`` controls the luminance filtering strength; ``hColor`` is kept small so
    chroma is lightly cleaned but not flattened. Mild defaults avoid blurring
    glyph edges — the classic pitfall of pre-OCR denoising.
    """
    h = max(1, int(strength))
    if img.ndim == 2:
        return cv2.fastNlMeansDenoising(img, None, h=h)
    return _from_bgr(
        cv2.fastNlMeansDenoisingColored(_to_bgr(img), None, h=h, hColor=max(1, h // 2))
    )


def apply_sharpen(
    img: np.ndarray, *, amount: float = 0.6, sigma: float = 1.0
) -> np.ndarray:
    """Unsharp mask: ``out = img + amount * (img - gaussian_blur)``.

    ``amount`` is the sharpening gain (keep ≤ ~1 for OCR to avoid ringing on
    thin strokes); ``sigma`` the blur radius. Operates on each channel.
    """
    ksize = 0  # let OpenCV size the kernel from sigma
    if img.ndim == 2:
        blur = cv2.GaussianBlur(img, (ksize, ksize), sigma)
        out = cv2.addWeighted(img, 1.0 + amount, blur, -amount, 0)
        return out
    blur = cv2.GaussianBlur(_to_bgr(img), (ksize, ksize), sigma)
    out = cv2.addWeighted(_to_bgr(img), 1.0 + amount, blur, -amount, 0)
    return _from_bgr(out)


def apply_threshold(img: np.ndarray, *, mode: str = "otsu") -> np.ndarray:
    """Binarize to a 3-channel image. ``mode`` ∈ {"otsu", "adaptive"}.

    Otsu picks a global threshold (uniform lighting); adaptive uses a local
    Gaussian window (uneven lighting / shadows). Output is expanded back to
    3 channels so the rest of the pipeline still sees RGB.
    """
    gray = img if img.ndim == 2 else cv2.cvtColor(_to_bgr(img), cv2.COLOR_BGR2GRAY)
    if mode == "adaptive":
        # INV so foreground (text) is black on white — matches typical scanned
        # text and the OCR engine's expectation. Block size must be odd & >1.
        bin_ = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 12
        )
    else:
        # Otsu returns the computed threshold; we use THRESH_BINARY+THRESH_OTSU.
        _, bin_ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(bin_, cv2.COLOR_GRAY2RGB)


def apply_morph_close(img: np.ndarray, *, ksize: int = 2) -> np.ndarray:
    """Morphological close (dilate then erode) to bridge fragmented strokes.

    Useful when thin/low-contrast text breaks apart; a small kernel closes
    small gaps without merging neighbouring glyphs.
    """
    k = max(1, int(ksize))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
    if img.ndim == 2:
        return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    out = cv2.morphologyEx(_to_bgr(img), cv2.MORPH_CLOSE, kernel)
    return _from_bgr(out)


# ---------------------------------------------------------------------------
# orchestrator
def enhance(img: np.ndarray, options: dict | None = None) -> np.ndarray:
    """Apply the enabled enhancement steps in a fixed order.

    ``options`` keys mirror :data:`_DEFAULTS`; missing keys fall back to the
    defaults. Returns a new RGB uint8 array (the input is never mutated).
    The order is: grayscale → CLAHE → denoise → sharpen → threshold →
    morph-close (see module docstring for rationale).
    """
    o = {**_DEFAULTS, **(options or {})}

    out = np.array(img, copy=True)

    if o.get("grayscale"):
        out = apply_grayscale_to_bgr(out)

    if o.get("clahe"):
        out = apply_clahe(
            out, clip=float(o.get("clahe_clip", 2.0)), tile=int(o.get("clahe_tile", 8))
        )

    if o.get("denoise"):
        out = apply_denoise(out, strength=int(o.get("denoise_strength", 7)))

    if o.get("sharpen"):
        out = apply_sharpen(
            out,
            amount=float(o.get("sharpen_amount", 0.6)),
            sigma=float(o.get("sharpen_sigma", 1.0)),
        )

    threshold_mode = o.get("threshold")
    if threshold_mode in ("otsu", "adaptive"):
        out = apply_threshold(out, mode=threshold_mode)

    if o.get("morph_close"):
        out = apply_morph_close(out, ksize=int(o.get("morph_ksize", 2)))

    return out
