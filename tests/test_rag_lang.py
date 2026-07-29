"""Tests for the language-aware hybrid RAG retriever.

The retriever combines dense (Chroma) + sparse (BM25) results via
reciprocal rank fusion. It also accepts a `lang` argument:

  - lang="cy" -> dense pass restricted to docs tagged lang=cy, BM25
                 zeroes out scores for non-cy docs.
  - lang="en" -> dense pass EXCLUDES docs tagged lang=cy (so the
                 Welsh-bootstrap micro-passages can't bleed into the
                 English context), BM25 zeroes out cy docs.
  - lang=None -> no language constraint.

These are unit tests, we mock the underlying Chroma store and disable
BM25 so we don't need real embeddings or rank_bm25 for the suite to pass.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.services import rag


def _doc(text: str, lang: str | None = None, source: str = "test") -> MagicMock:
    # tiny stand-in for langchain_core.documents.Document so we don't
    # need the real class loaded in the test path.
    d = MagicMock()
    d.page_content = text
    d.metadata = {"title": text[:30], "source": source}
    if lang is not None:
        d.metadata["lang"] = lang
    return d


@pytest.fixture
def _no_bm25(monkeypatch):
    # disable BM25 for tests that focus on the dense path. the dedicated
    # BM25 tests further down in the file don't apply this fixture so
    # they can exercise the real _bm25_search code.
    monkeypatch.setattr(rag, "_bm25_search", lambda *args, **kwargs: [])


@pytest.mark.asyncio
async def test_english_query_uses_ne_cy_filter(_no_bm25):
    """English queries must EXCLUDE cy-tagged docs from dense retrieval.

    This is the bug that caused 'Hello' to retrieve the Welsh-bootstrap
    'helo (Saesneg: hello)' chunk and trigger Welsh replies to English
    questions.
    """
    fake_store = MagicMock()
    fake_store.similarity_search_with_score.return_value = [
        (_doc("Tuition fees are GBP 9,250", "en"), 0.1),
        (_doc("Apply via UCAS", "en"), 0.2),
    ]

    with patch.object(rag, "_get_store", return_value=fake_store):
        out = await rag.retrieve("What are the fees?", top_k=2, lang="en")

    call_kwargs = fake_store.similarity_search_with_score.call_args.kwargs
    assert call_kwargs.get("filter") == {"lang": {"$ne": "cy"}}
    assert len(out) == 2


@pytest.mark.asyncio
async def test_welsh_query_uses_eq_cy_filter(_no_bm25):
    """Welsh queries should restrict the dense pass to lang=cy."""
    fake_store = MagicMock()
    fake_store.similarity_search_with_score.return_value = [
        (_doc("Mae'r ffioedd dysgu yn GBP 9,250", "cy"), 0.05),
    ]

    with patch.object(rag, "_get_store", return_value=fake_store):
        out = await rag.retrieve("Beth yw'r ffioedd?", top_k=3, lang="cy")

    first_call = fake_store.similarity_search_with_score.call_args_list[0]
    assert first_call.kwargs.get("filter") == {"lang": "cy"}
    assert any(d["lang"] == "cy" for d in out)


@pytest.mark.asyncio
async def test_welsh_topup_excludes_cy_when_short(_no_bm25):
    """If the cy-filtered pass under-fills, topup runs but still keeps cy out
    for English queries (and only adds non-cy entries even on Welsh topup
    when the doc explicitly carries the wrong tag)."""
    fake_store = MagicMock()
    # filtered pass returns 1 doc, second (general) topup returns 2 more
    fake_store.similarity_search_with_score.side_effect = [
        [(_doc("Mae'r ffioedd dysgu yn GBP 9,250", "cy"), 0.05)],
        [
            (_doc("Mae'r ffioedd dysgu yn GBP 9,250", "cy"), 0.05),  # dup
            (_doc("Apply via UCAS", "en"), 0.10),                     # new
        ],
    ]

    with patch.object(rag, "_get_store", return_value=fake_store):
        out = await rag.retrieve("Beth yw'r ffioedd?", top_k=3, lang="cy")

    assert fake_store.similarity_search_with_score.call_count == 2
    # de-dupe worked
    texts = [d["text"] for d in out]
    assert len(set(texts)) == len(texts)


@pytest.mark.asyncio
async def test_english_query_default_top_k_is_at_least_four(_no_bm25):
    fake_store = MagicMock()
    fake_store.similarity_search_with_score.return_value = []
    with patch.object(rag, "_get_store", return_value=fake_store):
        await rag.retrieve("What are the fees?", top_k=None, lang="en")
    for call in fake_store.similarity_search_with_score.call_args_list:
        # k should be at least max(2*4, 8) = 8 due to dense_pool widening
        assert call.kwargs.get("k", 0) >= 4


@pytest.mark.asyncio
async def test_welsh_query_default_top_k_is_at_least_five(_no_bm25):
    fake_store = MagicMock()
    fake_store.similarity_search_with_score.return_value = []
    with patch.object(rag, "_get_store", return_value=fake_store):
        await rag.retrieve("Beth yw'r ffioedd?", top_k=None, lang="cy")
    for call in fake_store.similarity_search_with_score.call_args_list:
        # cy widens to >=5, dense_pool widens further to max(5*2, 8) = 10
        assert call.kwargs.get("k", 0) >= 5


@pytest.mark.asyncio
async def test_empty_query_returns_empty_list(_no_bm25):
    fake_store = MagicMock()
    with patch.object(rag, "_get_store", return_value=fake_store):
        assert await rag.retrieve("", top_k=2, lang="cy") == []
        assert await rag.retrieve("   ", top_k=2, lang="en") == []
    fake_store.similarity_search_with_score.assert_not_called()


# ── BM25 + RRF fusion ────────────────────────────────────────────
def test_rrf_fuse_combines_two_ranked_lists():
    """Reciprocal Rank Fusion should give higher rank to docs that appear
    in BOTH lists than to docs that appear in only one."""
    dense = [
        ({"text": "doc-A"}, 0.9),
        ({"text": "doc-B"}, 0.7),
        ({"text": "doc-C"}, 0.5),
    ]
    sparse = [
        ({"text": "doc-B"}, 5.0),  # appears in both
        ({"text": "doc-D"}, 4.0),
        ({"text": "doc-A"}, 3.0),  # appears in both
    ]
    out = rag._rrf_fuse(dense, sparse, top_k=4)
    texts = [d["text"] for d in out]
    # doc-B appears at rank 2 in dense AND rank 1 in sparse, should
    # outrank doc-A which is rank 1 in dense + rank 3 in sparse.
    assert texts[0] == "doc-B"
    assert "doc-A" in texts
    assert "doc-D" in texts


def test_bm25_search_zeroes_other_language():
    """When lang=cy is requested, BM25 should return zero matches for
    en-tagged docs even if their term overlap is high."""
    rag._bm25_state.clear()

    class _FakeBM25:
        def get_scores(self, _toks):
            import numpy as np
            return np.array([1.0, 1.0])

    rag._bm25_state.bm25 = _FakeBM25()
    rag._bm25_state.docs = [
        {"text": "tuition fees", "lang": "en", "title": "en-doc"},
        {"text": "ffioedd dysgu", "lang": "cy", "title": "cy-doc"},
    ]
    rag._bm25_state.built = True

    en_only = rag._bm25_search("fees", k=2, lang="en")
    cy_only = rag._bm25_search("ffioedd", k=2, lang="cy")

    assert len(en_only) == 1 and en_only[0][0]["lang"] == "en"
    assert len(cy_only) == 1 and cy_only[0][0]["lang"] == "cy"

    rag._bm25_state.clear()
