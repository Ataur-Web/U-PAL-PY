"""tests for wellbeing signposting.

the ethics approval for this project commits to signposting external support
whenever a participant raises a wellbeing topic, on the basis that the study
carries a mild risk of distress. these tests exist so that commitment cannot
be broken silently by a later change to retrieval or prompting.

the signpost is appended deterministically after generation rather than being
left to the model, so these assertions are about the route, not the llm.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.routes import chat as chat_route
from app.services import llm


client = TestClient(app)


@pytest.fixture(autouse=True)
def _fresh_llm_client():
    # see test_sources.py - the chat client is a process-wide singleton bound
    # to the loop that created it, and TestClient makes a new loop per request
    llm._client = None
    llm._active_provider = "unknown"
    yield
    llm._client = None
    llm._active_provider = "unknown"


# the four services named on the approved ethics form, plus the uwtsd address.
# if any of these strings change, the form has to change too.
REQUIRED_EN = ["111", "116 123", "0808 164 0123", "85258", "08000 562 561"]
REQUIRED_CY = ["111", "116 123", "0808 164 0123", "85258", "08000 562 561"]


def test_signpost_fires_on_wellbeing_topic():
    assert chat_route._needs_signposting("I need help with health", None, "neutral")
    assert chat_route._needs_signposting("wellbeing support please", None, "neutral")
    assert chat_route._needs_signposting("I need counselling", None, "neutral")


def test_signpost_fires_on_distress_emotion_whatever_the_topic():
    """a student can disclose distress while asking about something else."""
    assert chat_route._needs_signposting("I can't cope with this", None, "distressed")
    assert chat_route._needs_signposting("my deadline is tomorrow", None, "stressed")


def test_signpost_fires_on_wellbeing_intent():
    for tag in chat_route._WELLBEING_TAGS:
        assert chat_route._needs_signposting("anything", {"tag": tag}, "neutral"), tag


def test_signpost_does_not_fire_on_ordinary_queries():
    """it should not appear on admin questions, or it becomes noise students
    learn to skip past."""
    for msg in (
        "What courses are available?",
        "Where is the SA1 campus?",
        "How much are the tuition fees?",
        "Thanks",
    ):
        assert not chat_route._needs_signposting(msg, None, "neutral"), msg


def test_all_ethics_form_services_present_in_both_languages():
    for token in REQUIRED_EN:
        assert token in chat_route._SIGNPOST_EN, f"missing from EN signpost: {token}"
    for token in REQUIRED_CY:
        assert token in chat_route._SIGNPOST_CY, f"missing from CY signpost: {token}"


def test_wellbeing_reply_carries_the_signpost_end_to_end():
    r = client.post("/api/chat", json={
        "message": "I need help with my wellbeing",
        "runningLang": "en",
        "history": [],
    })
    assert r.status_code == 200
    reply = r.json()["reply"]
    for token in REQUIRED_EN:
        assert token in reply, f"signpost missing {token!r} from live reply"


def test_welsh_wellbeing_reply_carries_the_welsh_signpost():
    r = client.post("/api/chat", json={
        "message": "Dw i angen cymorth lles",
        "runningLang": "cy",
        "history": [],
    })
    assert r.status_code == 200
    body = r.json()
    assert body["lang"] == "cy"
    for token in REQUIRED_CY:
        assert token in body["reply"], f"signpost missing {token!r} from live cy reply"


def test_ordinary_query_reply_has_no_signpost():
    r = client.post("/api/chat", json={
        "message": "Where is the SA1 campus?",
        "runningLang": "en",
        "history": [],
    })
    assert r.status_code == 200
    assert "116 123" not in r.json()["reply"]
