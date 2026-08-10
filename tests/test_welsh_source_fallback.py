"""Tests for English institutional fallback on Welsh answers.

UWTSD publishes far less Welsh than English: only 322 of the 2,835 indexed
corpus chunks are Welsh. A Welsh query is therefore usually grounded in the
BydTermCymru terminology layer, and the student was offered three dictionary
glosses - "cofrestru (Saesneg: register)" - in place of any page they could
act on.

Where the Welsh pass yields too few institutional pages, a second retrieval
runs over the English side and its pages are appended for attribution only.
The generation context stays Welsh, so the reply itself does not drift.
"""
from __future__ import annotations

import pytest

from app.routes import chat as chat_route


UWTSD = "https://www.uwtsd.ac.uk"
TERMAU = "https://termau.cymru/"


def _p(source: str, lang: str = "cy", text: str = "passage text here") -> dict:
    return {"text": text, "title": "t", "source": source, "lang": lang, "score": 0.5}


def test_is_institutional_distinguishes_the_two_source_kinds():
    assert chat_route._is_institutional(_p(f"{UWTSD}/swansea"))
    assert not chat_route._is_institutional(_p(TERMAU))
    assert not chat_route._is_institutional(_p(""))


@pytest.mark.asyncio
async def test_topup_adds_english_pages_when_welsh_has_none(monkeypatch):
    welsh_only = [_p(TERMAU), _p(TERMAU), _p(TERMAU)]

    async def fake_retrieve(query, top_k=None, lang=None):
        assert lang == "en", "the top-up must probe the English side"
        return [
            _p(f"{UWTSD}/study/how-apply", lang="en"),
            _p(f"{UWTSD}/apply/undergraduate", lang="en"),
        ]

    monkeypatch.setattr(chat_route.rag, "retrieve", fake_retrieve)
    out = await chat_route._topup_english_sources("how to register", welsh_only)

    # Welsh passages are preserved for the model, English pages appended
    assert out[:3] == welsh_only
    added = [p for p in out if chat_route._is_institutional(p)]
    assert len(added) == 2, "both available English pages were appended"
    assert all(p["lang"] == "en" for p in added)


@pytest.mark.asyncio
async def test_topup_skipped_when_welsh_already_has_institutional_pages(monkeypatch):
    already = [
        _p(f"{UWTSD}/cy/astudio"),
        _p(f"{UWTSD}/cy/ymchwil"),
        _p(f"{UWTSD}/cy/profiad-chyfleusterau"),
        _p(TERMAU),
    ]

    async def fail(*a, **kw):
        raise AssertionError("top-up should not have run")

    monkeypatch.setattr(chat_route.rag, "retrieve", fail)
    assert await chat_route._topup_english_sources("q", already) == already


@pytest.mark.asyncio
async def test_topup_failure_preserves_welsh_sources(monkeypatch):
    """A failed English probe must never cost the student the Welsh sources."""
    welsh_only = [_p(TERMAU), _p(TERMAU)]

    async def boom(*a, **kw):
        raise RuntimeError("chroma down")

    monkeypatch.setattr(chat_route.rag, "retrieve", boom)
    assert await chat_route._topup_english_sources("q", welsh_only) == welsh_only


def test_select_sources_excludes_terminology_entries():
    """BydTermCymru is a comprehension aid, not provenance. A gloss such as
    "cofrestru (Saesneg: register)" tells a student nothing about where the
    answer came from, so it must never appear as a source."""
    passages = [
        _p(TERMAU, text="gloss one"),
        _p(TERMAU, text="gloss two"),
        _p(TERMAU, text="gloss three"),
        _p(f"{UWTSD}/study/how-apply", lang="en", text="how to apply"),
        _p(f"{UWTSD}/apply/undergraduate", lang="en", text="undergraduate"),
    ]
    chosen = chat_route._select_sources(passages, limit=3)
    assert len(chosen) == 2, "only the two institutional pages qualify"
    assert all(chat_route._is_institutional(p) for p in chosen)


def test_select_sources_deduplicates_by_url():
    """Retrieval often returns two chunks of one page; they previously
    rendered as two identical-looking cards."""
    passages = [
        _p(f"{UWTSD}/swansea", text="chunk one"),
        _p(f"{UWTSD}/swansea", text="chunk two"),
        _p(f"{UWTSD}/study", text="another page"),
    ]
    chosen = chat_route._select_sources(passages, limit=3)
    assert len(chosen) == 2
    assert {p["source"] for p in chosen} == {f"{UWTSD}/swansea", f"{UWTSD}/study"}


def test_select_sources_may_return_nothing():
    """Showing no sources is preferable to padding the panel with glosses."""
    assert chat_route._select_sources([_p(TERMAU), _p(TERMAU)], limit=3) == []


def test_category_handles_trailing_slash():
    """A URL ending in a slash split to an empty string and rendered a blank
    badge, which is what the terminology sources displayed."""
    assert chat_route._category_for(f"{UWTSD}/study/how-apply", "en") == "How Apply"
    assert chat_route._category_for(f"{UWTSD}/swansea/", "en") == "Swansea"
    assert chat_route._category_for(None, "en") == "Information"
    assert chat_route._category_for(None, "cy") == "Gwybodaeth"
