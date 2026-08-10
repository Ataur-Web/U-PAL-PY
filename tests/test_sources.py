"""Chat endpoint sources attribution test — verifies that the response includes
sources with title, category, excerpt, and language fields."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services import llm


client = TestClient(app)


@pytest.fixture(autouse=True)
def _fresh_llm_client():
    """Drop the cached chat client between tests.

    llm._client is a module-level singleton, which is correct under uvicorn
    where the whole process shares one long-lived event loop. TestClient
    however spins up a new event loop per request, so a client cached during
    the first test holds an httpx connection pool bound to a loop that has
    since closed, and the next test fails with "Event loop is closed". Reset
    the singleton so each test builds its own client.
    """
    llm._client = None
    llm._active_provider = "unknown"
    yield
    llm._client = None
    llm._active_provider = "unknown"


def test_chat_sources_shape():
    """Verify that /api/chat returns sources with expected fields."""
    r = client.post("/api/chat", json={
        "message": "What courses does UWTSD offer?",
        "runningLang": "en",
    })
    assert r.status_code == 200
    body = r.json()

    # Check top-level response shape
    for key in ("reply", "lang", "sources"):
        assert key in body, f"Missing key: {key}"

    # Check that sources is a list
    assert isinstance(body["sources"], list)

    # If sources exist, check their shape
    if body["sources"]:
        for source in body["sources"]:
            for key in ("title", "category", "excerpt", "language"):
                assert key in source, f"Missing source field: {key}"
            # Verify types
            assert isinstance(source["title"], str)
            assert isinstance(source["category"], str)
            assert isinstance(source["excerpt"], str)
            assert source["language"] in ("en", "cy", "")
            # url is optional
            if "url" in source:
                assert isinstance(source["url"], (str, type(None)))

        # Verify we only return top 3
        assert len(body["sources"]) <= 3


def test_chat_sources_excerpt_length():
    """Verify that excerpts are truncated to ~150 characters."""
    r = client.post("/api/chat", json={
        "message": "Tell me about student support",
        "runningLang": "en",
    })
    assert r.status_code == 200
    body = r.json()

    # Check that excerpts don't exceed 150 chars (plus some buffer for trailing text)
    for source in body["sources"]:
        assert len(source["excerpt"]) <= 200, \
            f"Excerpt too long: {len(source['excerpt'])} chars"
