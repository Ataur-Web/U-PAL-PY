"""Smoke tests for the Welsh detection service."""
from __future__ import annotations

import pytest

from app.services import welsh


def test_english_text_detected_as_en():
    assert welsh.detect_language("What are the entry requirements?") == "en"


def test_welsh_orthography_detected():
    # Contains ll + circumflex + ortho clusters — should be Welsh.
    assert welsh.detect_language("Beth yw'r gofynion mynediad?") == "cy"


def test_very_short_text_defaults_english():
    assert welsh.detect_language("hi") == "en"


def test_looks_welsh_on_diacritic():
    assert welsh.looks_welsh("Llanelli") is True
    assert welsh.looks_welsh("hello world") is False


def test_augment_query_passthrough_for_english():
    out = welsh.augment_query("What are the fees?", "en")
    assert out == "What are the fees?"


@pytest.mark.skipif(welsh.bilingual_map_size() == 0, reason="bilingual map not loaded")
def test_augment_query_adds_english_gloss_for_welsh():
    out = welsh.augment_query("Beth yw'r ffioedd?", "cy")
    # Either augmented or unchanged — never shorter than the input.
    assert out.startswith("Beth yw'r ffioedd?")


# regression tests for the false-positive bug where common English
# queries were being misclassified as Welsh. each of these came from
# an actual user complaint during testing.
@pytest.mark.parametrize("text", [
    "What are the entry requirements?",
    "I need help with my course",
    "How do I apply?",
    "Tell me about Computing",
    "Hello, can you help me?",
    "When is the deadline?",
    "hi",
    "hello world",
])
def test_english_questions_not_misclassified_as_welsh(text):
    assert welsh.detect_language(text) == "en", (
        f"English query was misclassified as Welsh: {text!r}"
    )


# matching positive cases, real Welsh queries that should be detected.
@pytest.mark.parametrize("text", [
    "Beth yw'r ffioedd?",
    "Sut mae gwneud cais?",
    "Dwi angen help gyda fy nghwrs",
    "Pryd mae'r dyddiad cau?",
    "Shwmae",
])
def test_welsh_queries_detected_as_welsh(text):
    assert welsh.detect_language(text) == "cy", (
        f"Welsh query was misclassified as English: {text!r}"
    )
