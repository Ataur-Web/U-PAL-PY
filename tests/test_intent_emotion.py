"""Smoke tests for intent classifier + emotion detector.

Intent tests will silently skip when knowledge.json hasn't been copied in
yet — they're structural sanity checks for when data is present.
"""
from __future__ import annotations

from app.services import emotion, intent


# ── Emotion ──────────────────────────────────────────────────
def test_emotion_neutral():
    assert emotion.detect("What are the opening times?") == "neutral"


def test_emotion_stressed():
    assert emotion.detect("I'm so overwhelmed by this dissertation") == "stressed"


def test_emotion_frustrated():
    assert emotion.detect("This is trash, nothing works") == "frustrated"


def test_emotion_confused():
    assert emotion.detect("I don't understand what this means") == "confused"


def test_emotion_excited():
    assert emotion.detect("I'm so excited, can't wait to start!") == "excited"


def test_emotion_inherits_from_history():
    # Current message is neutral, but last user turn was stressed.
    result = emotion.detect(
        "ok",
        history=[{"role": "user", "text": "I'm burnt out and overwhelmed"}],
    )
    assert result == "stressed"


# ── Intent (data-dependent) ───────────────────────────────────
def test_intent_loads_without_data_gracefully():
    # Even with no knowledge.json, classify should return None not crash.
    result = intent.classify("hello there")
    assert result is None or isinstance(result, dict)
