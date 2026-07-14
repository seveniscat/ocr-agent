"""Tests for the VLM provider config endpoints (GET/POST /config/vlm).

The POST path writes to a real .env file. To avoid polluting the project's real
.env, these tests monkeypatch ``app.envstore.ENV_PATH`` to a tmp file and clear
the settings cache between calls so each read sees the just-written values.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# envstore.mask
# ---------------------------------------------------------------------------


def test_mask_empty_returns_empty():
    from app.envstore import mask
    assert mask("") == ""


def test_mask_short_secret_fully_masked():
    from app.envstore import mask
    # ≤8 chars → no partial leak
    assert mask("abc") == "***"
    assert mask("12345678") == "***"


def test_mask_normal_shows_first2_last4():
    from app.envstore import mask
    assert mask("sk-abcd1234") == "sk***1234"
    # the masked form must never contain the middle of the secret
    masked = mask("sk-SECRET-MIDDLE-PART-1234")
    assert "SECRET" not in masked
    assert "MIDDLE" not in masked
    assert masked == "sk***1234"


# ---------------------------------------------------------------------------
# endpoint tests — isolated against a tmp .env
# ---------------------------------------------------------------------------


def _isolated_env(monkeypatch, tmp_path: Path) -> Path:
    """Point envstore + the settings loader at a tmp .env and clear caches."""
    from app import envstore
    from app.config import get_settings

    env = tmp_path / ".env"
    env.write_text("")  # start empty
    monkeypatch.setattr(envstore, "ENV_PATH", env)
    # The Settings model reads .env from CWD by default; override env_file so it
    # points at our tmp file. Easiest robust way: clear the cache and rebuild
    # Settings each time via an instance whose env_file = tmp. We patch
    # get_settings to always read the tmp file.
    import app.config as cfg

    def _fresh():
        s = cfg.Settings(_env_file=str(env))
        return s

    monkeypatch.setattr(cfg, "get_settings", _fresh)
    # main.py imported get_settings by name → patch it there too.
    import app.main as main
    monkeypatch.setattr(main, "get_settings", _fresh)
    monkeypatch.setattr(main, "_settings", _fresh)
    get_settings.cache_clear()
    return env


def test_get_vlm_config_masks_key(monkeypatch, tmp_path):
    _isolated_env(monkeypatch, tmp_path)
    from app.main import app

    # Seed a key directly via the store, then GET must NOT echo it raw.
    from app import envstore
    envstore.upsert("OCR_VLM_API_KEY", "sk-supersecret-XYZ-1234")
    envstore.upsert("OCR_VLM_MODEL", "qwen-vl-max")

    c = TestClient(app)
    r = c.get("/config/vlm")
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "qwen-vl-max"
    assert body["has_key"] is True
    # The raw secret must never appear anywhere in the response.
    assert "supersecret" not in r.text
    assert body["api_key_masked"] == "sk***1234"


def test_get_vlm_config_reports_no_key(monkeypatch, tmp_path):
    _isolated_env(monkeypatch, tmp_path)
    from app.main import app

    c = TestClient(app)
    body = c.get("/config/vlm").json()
    assert body["has_key"] is False
    assert body["api_key_masked"] == ""


def test_save_vlm_config_writes_base_url_and_model(monkeypatch, tmp_path):
    env = _isolated_env(monkeypatch, tmp_path)
    from app.main import app

    c = TestClient(app)
    r = c.post("/config/vlm", json={
        "base_url": "https://example.test/v1",
        "model": "qwen-vl-plus",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["base_url"] == "https://example.test/v1"
    assert body["model"] == "qwen-vl-plus"
    # Persisted to the .env file. set_key may quote values; read it back the way
    # dotenv does rather than asserting a raw substring, so this survives any
    # quoting style python-dotenv chooses.
    from dotenv import get_key
    assert get_key(str(env), "OCR_VLM_BASE_URL") == "https://example.test/v1"
    assert get_key(str(env), "OCR_VLM_MODEL") == "qwen-vl-plus"


def test_save_vlm_config_persists_key_and_masks_it(monkeypatch, tmp_path):
    env = _isolated_env(monkeypatch, tmp_path)
    from app.main import app

    c = TestClient(app)
    r = c.post("/config/vlm", json={"api_key": "sk-mysecretkey-9876"})
    assert r.status_code == 200
    # Key written to .env (set_key may quote; read it back via dotenv) ...
    from dotenv import get_key
    assert get_key(str(env), "OCR_VLM_API_KEY") == "sk-mysecretkey-9876"
    # ... but masked in the response.
    assert "mysecretkey" not in r.text
    assert r.json()["has_key"] is True
    assert r.json()["api_key_masked"] == "sk***9876"


def test_save_vlm_config_empty_key_does_not_overwrite(monkeypatch, tmp_path):
    _isolated_env(monkeypatch, tmp_path)
    from app.main import app
    from app import envstore

    # Seed an existing key.
    envstore.upsert("OCR_VLM_API_KEY", "sk-existing-aaaa-0000")

    c = TestClient(app)
    # Save a model change WITHOUT re-entering the key (api_key = null).
    r = c.post("/config/vlm", json={"model": "qwen-vl-max"})
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "qwen-vl-max"
    # The original key survived.
    assert body["has_key"] is True
    assert body["api_key_masked"] == "sk***0000"


def test_save_vlm_config_toggles_understand(monkeypatch, tmp_path):
    env = _isolated_env(monkeypatch, tmp_path)
    from app.main import app

    c = TestClient(app)
    r = c.post("/config/vlm", json={"understand_enabled": False})
    assert r.status_code == 200
    assert r.json()["understand_enabled"] is False
    from dotenv import get_key
    assert get_key(str(env), "OCR_UNDERSTAND_ENABLED") == "false"


def test_save_vlm_config_toggles_thinking(monkeypatch, tmp_path):
    env = _isolated_env(monkeypatch, tmp_path)
    from app.main import app

    c = TestClient(app)
    r = c.post("/config/vlm", json={"enable_thinking": True})
    assert r.status_code == 200
    assert r.json()["enable_thinking"] is True
    from dotenv import get_key
    assert get_key(str(env), "OCR_VLM_ENABLE_THINKING") == "true"


def test_save_vlm_config_persists_rec_confidence_fallback(monkeypatch, tmp_path):
    """The fallback threshold is editable from the UI and persisted to .env."""
    env = _isolated_env(monkeypatch, tmp_path)
    from app.main import app

    c = TestClient(app)
    r = c.post("/config/vlm", json={"rec_confidence_fallback": 0.8})
    assert r.status_code == 200
    # Echoed back in the response payload (what the UI re-reads after save).
    assert r.json()["rec_confidence_fallback"] == 0.8
    # Persisted to .env under the OCR_REC_CONFIDENCE_FALLBACK key.
    from dotenv import get_key
    assert get_key(str(env), "OCR_REC_CONFIDENCE_FALLBACK") == "0.8"


def test_save_vlm_config_persists_rec_confidence_drop(monkeypatch, tmp_path):
    """The drop threshold is editable from the UI and persisted to .env."""
    env = _isolated_env(monkeypatch, tmp_path)
    from app.main import app

    c = TestClient(app)
    r = c.post("/config/vlm", json={"rec_confidence_drop": 0.5})
    assert r.status_code == 200
    # Echoed back in the response payload (what the UI re-reads after save).
    assert r.json()["rec_confidence_drop"] == 0.5
    # Persisted to .env under the OCR_REC_CONFIDENCE_DROP key.
    from dotenv import get_key
    assert get_key(str(env), "OCR_REC_CONFIDENCE_DROP") == "0.5"


def test_save_vlm_config_refreshes_running_pipeline(monkeypatch, tmp_path):
    """Regression: a key saved through the UI must reach an already-running
    pipeline. Before the fix, Pipeline.settings was a snapshot taken at startup
    (before the key existed), so /understand kept reporting 503 even after a
    successful save. The fix hot-swaps pipeline.settings on save.
    """
    _isolated_env(monkeypatch, tmp_path)
    from app.main import app, _get_pipeline_existing

    # Build the pipeline once, with NO key — like a server started before config.
    # _get_pipeline_existing returns None until _get_pipeline() builds it; force it.
    from app.main import _get_pipeline
    pipeline = _get_pipeline()
    assert pipeline.settings.vlm_api_key == ""  # precondition: no key yet

    c = TestClient(app)
    # Save a key via the real endpoint (the path the UI takes).
    r = c.post("/config/vlm", json={"api_key": "sk-saved-runtime-9999"})
    assert r.status_code == 200

    # The SAME pipeline instance must now see the key, without restart.
    assert pipeline.settings.vlm_api_key == "sk-saved-runtime-9999"
    # And building the VLM from it must NOT raise "key not set".
    pipeline._vlm = None  # drop any cache so _get_vlm rebuilds
    try:
        vlm = pipeline._get_vlm()
        assert vlm is not None
    except RuntimeError as e:
        pytest.fail(f"VLM build still failed after save: {e}")
