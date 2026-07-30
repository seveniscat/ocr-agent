"""Tests for the POST /preprocess endpoint.

/preprocess does NOT touch the OCR pipeline (no PaddleOCR / VLM), so we only
need to patch _settings (used by _resolve_image for url timeouts). It decodes
bytes with cv2, runs app.imgenhance.enhance, and returns image/png.
"""
import io

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import main as main_mod
from app.config import Settings


def _png_bytes(size=(64, 64), color=(240, 240, 240)) -> bytes:
    """Tiny solid-colour PNG (no text) — enough to exercise the decode path."""
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def client(monkeypatch):
    # _resolve_image reads url-fetch timeouts from _settings(); patch so a
    # missing url won't try real network. (We only upload files here, but keep
    # it consistent with the rest of the suite.)
    monkeypatch.setattr(main_mod, "_settings", lambda: Settings())
    return TestClient(main_mod.app)


def test_preprocess_returns_png(client):
    r = client.post(
        "/preprocess",
        files={"file": ("x.png", _png_bytes(), "image/png")},
        data={"options": '{"clahe":true,"sharpen":true}'},
    )
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert len(r.content) > 0
    # Response must be a decodable image of the same size.
    arr = np.frombuffer(r.content, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    assert img is not None
    assert img.shape[:2] == (64, 64)


def test_preprocess_defaults_without_options(client):
    # No options field → server defaults (CLAHE + sharpen). Still a valid PNG.
    r = client.post("/preprocess", files={"file": ("x.png", _png_bytes(), "image/png")})
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"
    assert len(r.content) > 0


def test_preprocess_threshold_produces_binary(client):
    # A noisy gray image + otsu → output pixels are only 0/255.
    rng = np.random.default_rng(1).integers(150, 230, size=(40, 120)).astype(np.uint8)
    rng[15:25, 10:110] = 90
    img = np.stack([rng, rng, rng], axis=-1)
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")

    r = client.post(
        "/preprocess",
        files={"file": ("n.png", buf.getvalue(), "image/png")},
        data={"options": '{"clahe":false,"sharpen":false,"threshold":"otsu"}'},
    )
    assert r.status_code == 200
    arr = np.frombuffer(r.content, dtype=np.uint8)
    out = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    unique = set(int(u) for u in np.unique(out))
    assert unique.issubset({0, 255})


def test_preprocess_missing_image_is_400(client):
    # Neither file nor url → _resolve_image raises HTTPException(400).
    r = client.post("/preprocess", data={"options": "{}"})
    assert r.status_code == 400


def test_preprocess_garbage_bytes_is_400(client):
    # Decodable-looking multipart but non-image bytes → cv2.imdecode → None.
    r = client.post(
        "/preprocess",
        files={"file": ("x.png", b"not an image at all", "image/png")},
    )
    assert r.status_code == 400


def test_preprocess_bad_options_json_is_400(client):
    r = client.post(
        "/preprocess",
        files={"file": ("x.png", _png_bytes(), "image/png")},
        data={"options": "{not json"},
    )
    assert r.status_code == 400


def test_preprocess_accepts_url_form_field(client):
    # The endpoint signature accepts `url`; without network we only assert the
    # route accepts the field and reaches _resolve_image (which 400s on a bad
    # scheme rather than crashing). Confirms the param is wired.
    r = client.post("/preprocess", data={"url": "ftp://example/x.png"})
    assert r.status_code == 400
