"""Unit tests for app.imgenhance — the pure-function enhancement steps.

These build small synthetic RGB arrays with numpy (no PIL needed) and assert
the invariants each step must uphold: same shape/dtype, valid value range, and
the specific behaviour of each operator (e.g. threshold → only 0/255, grayscale
→ channels identical). enhance() chaining is covered too.
"""
import numpy as np
import pytest

from app.imgenhance import (
    apply_clahe,
    apply_denoise,
    apply_grayscale_to_bgr,
    apply_morph_close,
    apply_sharpen,
    apply_threshold,
    enhance,
)


def _white_with_text(h=40, w=120):
    """White background with a few near-black "text" rows → realistic for OCR."""
    img = np.full((h, w, 3), 255, dtype=np.uint8)
    img[14:26, 10:110] = 20  # a dark text block
    return img


def _noisy_gray(h=40, w=120, low=180):
    """Low-contrast noisy gray field (e.g. faded scan) to exercise CLAHE."""
    rng = np.random.default_rng(0)
    base = rng.integers(low, 230, size=(h, w)).astype(np.int16)
    base[14:26, 10:110] -= 60  # darker "text" block, same value on all channels
    img = np.stack([base, base, base], axis=-1).clip(0, 255).astype(np.uint8)
    return img


# ---------------------------------------------------------------------------
# individual steps: shape / dtype / range invariants
@pytest.mark.parametrize("img", [_white_with_text(), _noisy_gray()])
def test_clahe_preserves_shape_dtype_range(img):
    out = apply_clahe(img, clip=2.0, tile=8)
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    assert out.min() >= 0 and out.max() <= 255


def test_clahe_on_single_channel_returns_2d():
    gray = _white_with_text()[:, :, 0]
    out = apply_clahe(gray)
    assert out.ndim == 2
    assert out.shape == gray.shape


def test_sharpen_preserves_shape_and_range():
    img = _white_with_text()
    out = apply_sharpen(img, amount=0.6)
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    assert out.min() >= 0 and out.max() <= 255


def test_denoise_preserves_shape():
    img = _noisy_gray()
    out = apply_denoise(img, strength=7)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


def test_grayscale_collapses_channels_to_identical():
    img = _white_with_text()
    out = apply_grayscale_to_bgr(img)
    assert out.shape == img.shape  # still 3-channel
    # All three channels must be identical after grayscale.
    assert np.array_equal(out[:, :, 0], out[:, :, 1])
    assert np.array_equal(out[:, :, 1], out[:, :, 2])


@pytest.mark.parametrize("mode", ["otsu", "adaptive"])
def test_threshold_is_binary(mode):
    img = _noisy_gray()
    out = apply_threshold(img, mode=mode)
    assert out.shape == img.shape
    unique = np.unique(out)
    assert set(int(u) for u in unique).issubset({0, 255})


def test_morph_close_preserves_shape():
    img = _white_with_text()
    out = apply_morph_close(img, ksize=2)
    assert out.shape == img.shape
    assert out.dtype == np.uint8


# ---------------------------------------------------------------------------
# enhance() orchestration
def test_enhance_empty_options_is_identity():
    img = _white_with_text()
    out = enhance(img, {k: False for k in [
        "clahe", "sharpen", "denoise", "grayscale", "morph_close"
    ]} | {"threshold": None})
    # With every step off, output must equal input byte-for-byte.
    assert np.array_equal(out, img)


def test_enhance_does_not_mutate_input():
    img = _white_with_text()
    snapshot = img.copy()
    enhance(img, {"clahe": True, "sharpen": True})
    assert np.array_equal(img, snapshot)


def test_enhance_defaults_keep_shape_and_enhance():
    img = _noisy_gray(low=190)
    out = enhance(img)  # defaults: CLAHE + sharpen on
    assert out.shape == img.shape
    assert out.dtype == np.uint8
    # CLAHE should widen the contrast range of a low-contrast input.
    assert int(out.max()) - int(out.min()) >= int(img.max()) - int(img.min())


def test_enhance_threshold_chain_produces_binary():
    img = _noisy_gray()
    out = enhance(img, {"clahe": False, "sharpen": False, "threshold": "otsu"})
    assert set(int(u) for u in np.unique(out)).issubset({0, 255})


def test_enhance_full_pipeline_runs_without_error():
    img = _white_with_text()
    out = enhance(img, {
        "grayscale": True, "clahe": True, "denoise": True, "sharpen": True,
        "threshold": "adaptive", "morph_close": True,
    })
    assert out.shape == img.shape
    assert out.dtype == np.uint8
