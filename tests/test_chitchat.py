"""Tests for conversational-turn handling.

Regression cover for a bug where a closing message such as "Thanks", sent
after a question about courses, was answered with a full course listing.

Three things combined to cause it:

1. The scraped corpus contained degenerate chunks - fragments such as ";" and
   ", MSc" left behind by text extraction. A near-empty string embeds to a
   vector that sits spuriously close to any short query, so these were the top
   dense hit for "thanks", "wow" and "ok" alike.
2. Retrieval always returns its top-k however poor the matches are, so those
   fragments were passed to the model as UWTSD context.
3. Given institutional passages in its prompt, an 8B model talks about them.

The fix filters degenerate chunks at ingestion and gates retrieval on relative
dense distance. Generation is deliberately NOT short-circuited: the LLM runs on
every turn so it can respond to the conversation in context.
"""
from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app
from app.routes import chat as chat_route
from app.services import intent, llm


client = TestClient(app)


COURSE_HISTORY = [
    {"role": "user", "text": "What courses are available?"},
    {"role": "assistant",
     "text": "We have a wide range of courses across various subjects at UWTSD, "
             "including Art and Design, Business, Computing, Education and Nursing."},
]


import pytest


@pytest.fixture(autouse=True)
def _fresh_llm_client():
    # llm._client is a process-wide singleton bound to the event loop that
    # created it; TestClient makes a new loop per request. see test_sources.py
    llm._client = None
    llm._active_provider = "unknown"
    yield
    llm._client = None
    llm._active_provider = "unknown"


def test_history_boost_can_be_disabled():
    """With the boost on, prior course turns outrank the chit-chat intent.
    With it off, 'Thanks' must classify as conversational."""
    unboosted = intent.classify(
        "Thanks", "en", history=COURSE_HISTORY, use_history_boost=False,
    )
    assert unboosted is not None
    assert unboosted["tag"] in chat_route._CHITCHAT_TAGS


def test_real_questions_never_take_the_conversational_path():
    """Regression: 'Are there any courses available?' scored 76.9 against a
    greeting pattern on the shared tokens 'are' and 'there', which suppressed
    retrieval and left a genuine question answered from nothing."""
    for msg in (
        "What courses are available?",
        "Are there any courses available?",
        "Tell me about student support",
        "How do I apply?",
        "Where is the SA1 campus?",
    ):
        assert not chat_route._is_conversational(msg), f"{msg!r} misrouted"


def test_short_greetings_do_take_the_conversational_path():
    for msg in ("Thanks", "hello", "cheers mate", "Diolch yn fawr"):
        assert chat_route._is_conversational(msg), f"{msg!r} misrouted"


def test_short_but_substantive_message_is_not_conversational():
    """'fees?' is one word but names a topic, so it must still retrieve."""
    assert not chat_route._is_conversational("fees?")
    assert not chat_route._is_conversational("llety?")


def test_relevance_filter_drops_the_noise_tail():
    """Passages far worse than the best match are dropped; BM25-only passages
    (no dense score) are kept."""
    passages = [
        {"text": "best", "score": 0.50},
        {"text": "close enough", "score": 0.70},
        {"text": "far tail", "score": 1.21},
        {"text": "keyword hit", "score": None},
    ]
    kept = [p["text"] for p in chat_route._filter_by_relevance(passages)]
    assert "best" in kept
    assert "close enough" in kept
    assert "keyword hit" in kept
    assert "far tail" not in kept


def test_relevance_filter_drop_all():
    passages = [{"text": "anything", "score": 0.5}]
    assert chat_route._filter_by_relevance(passages, drop_all=True) == []


def test_relevance_filter_passes_through_when_no_scores():
    passages = [{"text": "a", "score": None}, {"text": "b", "score": None}]
    assert len(chat_route._filter_by_relevance(passages)) == 2


def test_thanks_after_course_question_does_not_return_course_answer():
    """The reported bug: 'Thanks' following a course question came back with a
    course listing. The reply is still LLM-generated, but must not carry the
    previous topic over, and must cite nothing."""
    r = client.post("/api/chat", json={
        "message": "Thanks",
        "runningLang": "en",
        "history": COURSE_HISTORY,
    })
    assert r.status_code == 200
    body = r.json()

    assert body["reply"].strip(), "must still produce a generated reply"
    assert body["sources"] == [], "conversational turns cite nothing"
    reply = body["reply"].lower()
    for leaked in ("undergraduate", "postgraduate", "art and design"):
        assert leaked not in reply, f"previous topic leaked into reply: {leaked!r}"


def test_thank_you_mid_conversation_does_not_restate_the_last_answer():
    """Regression: "Thank you very much", sent after an accommodation answer,
    came back as the whole accommodation explanation again (533 chars).

    Conversational turns are answered with no history at all, so there is
    nothing to restate and nothing left over to contaminate later turns.
    """
    history = [
        {"role": "user", "text": "How do i apply for accomodation?"},
        {"role": "assistant", "text":
            "Once you've secured your place at UWTSD, you'll be able to apply for "
            "accommodation through the university's halls system. The key thing is "
            "to apply early — availability can change, so it's worth getting your "
            "application in as soon as you've confirmed your place. You might also "
            "want to check out UniKitOut for student essentials."},
    ]
    r = client.post("/api/chat", json={
        "message": "Thank you very much",
        "runningLang": "en",
        "history": history,
    })
    assert r.status_code == 200
    body = r.json()

    assert body["sources"] == []
    reply = body["reply"]
    assert reply.strip(), "must still be LLM-generated, not blank"
    # a thank-you is answered in a sentence or two, not a re-explanation
    assert len(reply) < 300, f"reply restated the previous answer ({len(reply)} chars)"
    lowered = reply.lower()
    for leaked in ("halls", "unikitout", "apply early", "accommodation portal"):
        assert leaked not in lowered, f"previous answer leaked: {leaked!r}"


def test_real_question_still_uses_retrieval():
    """The relevance gate must not swallow genuine information requests."""
    r = client.post("/api/chat", json={
        "message": "Where is the SA1 campus?",
        "runningLang": "en",
        "history": COURSE_HISTORY,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["sources"], "a factual question should still return sources"


def test_sources_are_never_empty_fragments():
    """Degenerate chunks such as ';' must not reach the source panel."""
    r = client.post("/api/chat", json={
        "message": "How much are the tuition fees?",
        "runningLang": "en",
        "history": [],
    })
    assert r.status_code == 200
    for source in r.json()["sources"]:
        excerpt = source["excerpt"].strip()
        letters = sum(1 for c in excerpt if c.isalnum())
        assert letters >= 10, f"degenerate excerpt surfaced as a source: {excerpt!r}"
