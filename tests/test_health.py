"""Health endpoint contract test — doesn't depend on Ollama or Chroma
being live; just verifies the route wires up and returns the expected keys."""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert body["service"] == "U-PAL-PY"


def test_health_shape():
    r = client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    for key in ("status", "ollama", "ollamaModel", "chroma", "welsh",
                "bilingualTerms", "welshVocab"):
        assert key in body
    assert body["status"] == "OK"
