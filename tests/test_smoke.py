"""Smoke test: the FastAPI app imports and /healthz responds.

Does NOT exercise the OCR engine (no paddle dependency required to run).
The OCR pipeline itself is integration-tested separately.
"""
from __future__ import annotations

import pytest

pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


def test_healthz():
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    resp = client.get("/healthz")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_unknown_task_returns_404():
    from fastapi.testclient import TestClient

    from app.main import app

    client = TestClient(app)
    resp = client.get("/tasks/does-not-exist")
    assert resp.status_code == 404
