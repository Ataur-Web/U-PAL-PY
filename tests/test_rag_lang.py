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
from langchain_core.documents import Document

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
    # disable the sparse pass for tests that focus on the dense path. with no
    # sparse retriever, retrieve() degrades to dense-only, so these tests
    # observe the Chroma filter directly. the dedicated sparse and fusion
    # tests further down don't apply this fixture.
    monkeypatch.setattr(rag, "_get_sparse_retriever", lambda *args, **kwargs: None)


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
    # the cy exclusion is paired with the conversational-intent exclusion,
    # which applies in both languages. see rag._CONVERSATIONAL_TITLES.
    assert call_kwargs.get("filter") == {
        "$and": [
            {"lang": {"$ne": "cy"}},
            {"title": {"$nin": rag._CONVERSATIONAL_TITLES}},
        ]
    }
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
    # cy is constrained to Welsh passages AND excludes the BydTermCymru
    # glosses, which are 84% of the Welsh slice and were taking over half the
    # dense top-5 slots. they remain available to the BM25 pass.
    # ...and the conversational intents, which crowded the Welsh ranking for
    # the same reason: the cy content slice is small, so a handful of short
    # greeting strings is a large share of the candidate pool.
    assert first_call.kwargs.get("filter") == {
        "$and": [
            {"lang": "cy"},
            {"source": {"$ne": "https://termau.cymru/"}},
            {"title": {"$nin": rag._CONVERSATIONAL_TITLES}},
        ]
    }
    assert any(d["lang"] == "cy" for d in out)


def test_sparse_bucket_drops_conversational_intents(monkeypatch):
    """The BM25 pool must exclude the conversational intents too.

    The dense filter alone is not enough: a greeting shares common function
    words with short Welsh questions ("sut", "am", "i"), so BM25 ranked
    how_are_you above the curated fact for several benchmark queries. Unlike
    the BydTermCymru glosses, which BM25 keeps because exact term overlap is
    what a glossary is for, a greeting has no terms worth matching.
    """
    pool = [
        Document(page_content="Shwmae! Croeso i U-Pal",
                 metadata={"title": "greeting", "lang": "cy"}),
        Document(page_content="Mae oriau agor llyfrgelloedd PCYDDS yn amrywio",
                 metadata={"title": "Library Hours", "lang": "cy"}),
    ]
    monkeypatch.setattr(rag, "_collection_documents", lambda: pool)
    rag._sparse_cache.clear()
    try:
        retriever = rag._get_sparse_retriever("cy", 5)
        assert retriever is not None, "rank_bm25 unavailable"
        titles = [d.metadata.get("title") for d in retriever.invoke("oriau agor y llyfrgell")]
        assert "greeting" not in titles
        assert "Library Hours" in titles
    finally:
        # the cache is process-wide, so leave it clean for later tests
        rag._sparse_cache.clear()


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


# ── BM25 bucketing + EnsembleRetriever fusion ────────────────────
def test_sparse_retriever_buckets_by_language(monkeypatch):
    """BM25 has no metadata filter, so the language boundary is enforced by
    building one index per bucket. cy takes only cy docs; en excludes cy but
    keeps untagged docs, mirroring the Chroma $ne filter."""
    docs = [
        Document(page_content="tuition fees for undergraduate study",
                 metadata={"lang": "en", "title": "en-doc"}),
        Document(page_content="ffioedd dysgu israddedig",
                 metadata={"lang": "cy", "title": "cy-doc"}),
        Document(page_content="an untagged passage mentioning fees",
                 metadata={"title": "untagged-doc"}),
    ]
    rag._sparse_cache.clear()
    monkeypatch.setattr(rag, "_collection_documents", lambda: docs)

    cy_retriever = rag._get_sparse_retriever("cy", k=3)
    en_retriever = rag._get_sparse_retriever("en", k=3)

    assert [d.metadata.get("lang") for d in cy_retriever.docs] == ["cy"]
    # en bucket keeps the en doc and the untagged doc, drops the cy one
    en_langs = [d.metadata.get("lang") for d in en_retriever.docs]
    assert "cy" not in en_langs
    assert len(en_retriever.docs) == 2

    rag._sparse_cache.clear()


def test_sparse_retriever_returns_none_for_empty_bucket(monkeypatch):
    """An empty language bucket must degrade to dense-only, not crash."""
    rag._sparse_cache.clear()
    monkeypatch.setattr(rag, "_collection_documents", lambda: [])
    assert rag._get_sparse_retriever("cy", k=3) is None
    rag._sparse_cache.clear()


@pytest.mark.asyncio
async def test_ensemble_fuses_dense_and_sparse_with_rrf_constant():
    """Both retrievers must reach EnsembleRetriever, which fuses them with
    reciprocal rank fusion at the canonical c=60."""
    fake_store = MagicMock()
    fake_store.similarity_search_with_score.return_value = [
        (_doc("Tuition fees are GBP 9,250", "en"), 0.1),
    ]
    fake_sparse = MagicMock()
    captured: dict = {}

    class _FakeEnsemble:
        def __init__(self, retrievers, weights, c):
            captured["retrievers"] = retrievers
            captured["weights"] = weights
            captured["c"] = c

        def invoke(self, _query):
            return [_doc("Tuition fees are GBP 9,250", "en")]

    with patch.object(rag, "_get_store", return_value=fake_store), \
         patch.object(rag, "_get_sparse_retriever", return_value=fake_sparse), \
         patch.object(rag, "EnsembleRetriever", _FakeEnsemble):
        out = await rag.retrieve("What are the fees?", top_k=2, lang="en")

    assert captured["c"] == 60
    assert captured["weights"] == [0.5, 0.5]
    assert len(captured["retrievers"]) == 2
    assert captured["retrievers"][1] is fake_sparse
    assert out and out[0]["text"].startswith("Tuition fees")


@pytest.mark.asyncio
async def test_falls_back_to_dense_when_ensemble_unavailable(monkeypatch):
    """If EnsembleRetriever is missing, retrieval must still return dense
    results rather than failing the request."""
    fake_store = MagicMock()
    fake_store.similarity_search_with_score.return_value = [
        (_doc("Apply via UCAS", "en"), 0.2),
    ]
    monkeypatch.setattr(rag, "EnsembleRetriever", None)
    monkeypatch.setattr(rag, "_get_sparse_retriever", lambda *a, **kw: MagicMock())

    with patch.object(rag, "_get_store", return_value=fake_store):
        out = await rag.retrieve("How do I apply?", top_k=2, lang="en")

    assert out and out[0]["text"] == "Apply via UCAS"
