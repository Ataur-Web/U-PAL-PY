"""ChromaDB retriever with BM25 hybrid search.

stores the UWTSD corpus, knowledge base, OpenOrca, Natural Questions and
Welsh-chat passages as a single Chroma collection. on top of dense
similarity from sentence-transformers we run a BM25 sparse retriever
and fuse the two ranked lists with reciprocal rank fusion.

public functions:
    retrieve(query, top_k, lang)   used by the /api/chat route
    ingest_documents(docs)         called by scripts/ingest.py and the fetchers
    rebuild_bm25()                 force the in-memory BM25 index to refresh

we persist the dense Chroma collection to disk (CHROMA_PERSIST_DIR in
.env). the BM25 index is built in-memory at first use from the dense
collection and refreshed on every ingest_documents() call.

ref: https://docs.trychroma.com/
ref: Reimers, N. and Gurevych, I. (2019) Sentence-BERT, EMNLP 2019,
     for the multilingual MiniLM embedding model
ref: Robertson, S. and Zaragoza, H. (2009) The Probabilistic Relevance
     Framework: BM25 and Beyond, Foundations and Trends in IR
ref: Cormack, G.V., Clarke, C.L.A. and Buettcher, S. (2009) Reciprocal
     Rank Fusion outperforms Condorcet and individual rank learning
     methods, SIGIR 2009
"""
from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Any

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document

from app.config import get_settings


log = logging.getLogger("u-pal-py.rag")


# we cache the embedding model and Chroma store because each of them
# takes several seconds to initialise (loading the sentence-transformer
# weights and opening the sqlite file). lru_cache means the cost is paid
# once per process.
# ref: https://docs.python.org/3/library/functools.html#functools.lru_cache
@lru_cache(maxsize=1)
def _get_embeddings() -> HuggingFaceEmbeddings:
    s = get_settings()
    log.info("Loading embedding model: %s", s.embedding_model)
    # normalize_embeddings=True makes cosine similarity equivalent to dot
    # product, which Chroma's default uses internally.
    # ref: https://www.sbert.net/docs/usage/semantic_textual_similarity.html
    return HuggingFaceEmbeddings(
        model_name=s.embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )


@lru_cache(maxsize=1)
def _get_store() -> Chroma:
    s = get_settings()
    return Chroma(
        collection_name=s.chroma_collection,
        persist_directory=s.chroma_persist_dir,
        embedding_function=_get_embeddings(),
    )


# Chroma's underlying SQLite-backed upsert has a hard cap of 5461 docs
# per call (the limit is set in the chromadb rust binding and depends on
# SQLite's max parameter count). we batch internally so every caller
# can hand us an arbitrarily large list.
# ref: https://github.com/chroma-core/chroma/issues/1049
_CHROMA_MAX_BATCH = 5000


def ingest_documents(docs: list[Document]) -> int:
    # called by the ingest script and the fetch_* helpers. returns how
    # many docs we added so the CLI can print a summary.
    if not docs:
        return 0
    store = _get_store()
    total = len(docs)
    for start in range(0, total, _CHROMA_MAX_BATCH):
        chunk = docs[start:start + _CHROMA_MAX_BATCH]
        store.add_documents(chunk)
        log.info(
            "  ingested batch %d-%d of %d",
            start + 1, start + len(chunk), total,
        )
    log.info("Ingested %d documents into Chroma", total)
    # the BM25 index is now stale. invalidate so the next retrieve()
    # rebuilds it from the latest collection contents.
    _bm25_state.clear()
    return total


def reset_collection() -> None:
    # dev helper for when we change the embedding model and need to
    # re-embed everything from scratch.
    #
    # after delete_collection() the cached Chroma wrapper still holds a
    # handle to the (now deleted) collection, so the next add_documents
    # call would crash with "Chroma collection not initialized". we clear
    # the lru_cache so the next _get_store() rebuilds with a fresh
    # collection, which langchain_chroma auto-creates on construction.
    # ref: langchain_chroma vectorstores._collection guard
    store = _get_store()
    try:
        store.delete_collection()
        log.warning("Chroma collection dropped")
    except Exception as e:
        log.error("Failed to drop collection: %s", e)
    finally:
        _get_store.cache_clear()
        _bm25_state.clear()


def get_collection_count() -> int:
    # used by /api/health to show the doc count in the CONNECTION card.
    # _collection is a private attr but it's the only way to get the
    # count without doing a dummy query.
    try:
        store = _get_store()
        return store._collection.count()     # type: ignore[attr-defined]
    except Exception as e:
        log.warning("Chroma count failed: %s", e)
        return 0


# ── BM25 sparse retrieval ────────────────────────────────────────────
# we build the BM25 index lazily from the Chroma collection on first use.
# rank-bm25 holds tokenised documents in memory. for ~30k UWTSD docs
# this is roughly 30 MB of strings + lists, well within budget.
# ref: https://github.com/dorianbrown/rank_bm25
_TOKEN_RE = re.compile(r"[a-zâêîôûŵŷ0-9]+", re.IGNORECASE)


def _tokenise(text: str) -> list[str]:
    # lowercase + ascii-fold-ish word tokenisation. shared by index-build
    # and query-time so the term spaces match.
    if not text:
        return []
    return [t.lower() for t in _TOKEN_RE.findall(text)]


class _Bm25State:
    def __init__(self) -> None:
        self.bm25 = None                      # rank_bm25.BM25Okapi instance
        self.docs: list[dict[str, Any]] = []  # parallel list, one per indexed doc
        self.built: bool = False

    def clear(self) -> None:
        self.bm25 = None
        self.docs = []
        self.built = False


_bm25_state = _Bm25State()


def _build_bm25() -> None:
    # walk the entire Chroma collection once and feed every doc to BM25Okapi.
    # we do this lazily because the cold cost is ~2-5 s for 30k docs and
    # we don't want it on the import path.
    try:
        from rank_bm25 import BM25Okapi   # type: ignore[import]
    except ImportError:
        log.warning("rank_bm25 not installed, hybrid search disabled")
        _bm25_state.built = True
        return

    try:
        store = _get_store()
        # _collection.get() returns ALL documents. for 30k docs this is
        # roughly 60 ms.
        # ref: https://docs.trychroma.com/reference/Collection#get
        raw = store._collection.get(   # type: ignore[attr-defined]
            include=["documents", "metadatas"],
        )
    except Exception as e:
        log.warning("BM25 build failed at fetch step: %s", e)
        _bm25_state.built = True
        return

    docs_text = raw.get("documents") or []
    metas     = raw.get("metadatas")  or [{}] * len(docs_text)

    tokenised: list[list[str]] = []
    docs_meta: list[dict[str, Any]] = []
    for text, meta in zip(docs_text, metas):
        toks = _tokenise(text)
        if not toks:
            continue
        tokenised.append(toks)
        docs_meta.append({
            "text":  text,
            "title": (meta or {}).get("title", "corpus"),
            "source": (meta or {}).get("source", ""),
            "lang":  (meta or {}).get("lang", ""),
        })

    if not tokenised:
        log.info("BM25 index empty, no docs to index")
        _bm25_state.built = True
        return

    _bm25_state.bm25 = BM25Okapi(tokenised)
    _bm25_state.docs = docs_meta
    _bm25_state.built = True
    log.info("BM25 index built over %d docs", len(docs_meta))


def rebuild_bm25() -> None:
    # external entry point for when callers know the index is stale
    # (e.g. after a CLI ingest). also called automatically by
    # ingest_documents().
    _bm25_state.clear()
    _build_bm25()


def _bm25_search(query: str, k: int, lang: str | None) -> list[tuple[dict, float]]:
    # returns top-k BM25 matches as (doc_dict, score) tuples. higher score
    # = more relevant in BM25 (opposite of Chroma's distance score).
    if not _bm25_state.built:
        _build_bm25()
    if _bm25_state.bm25 is None or not _bm25_state.docs:
        return []

    toks = _tokenise(query)
    if not toks:
        return []

    scores = _bm25_state.bm25.get_scores(toks)

    # if a lang filter is requested, zero out the scores of docs in the
    # other language. we don't filter the index itself because that would
    # require rebuilding per-query.
    if lang in ("en", "cy"):
        for i, doc in enumerate(_bm25_state.docs):
            doc_lang = doc.get("lang", "")
            if lang == "cy" and doc_lang != "cy":
                scores[i] = 0.0
            elif lang == "en" and doc_lang == "cy":
                scores[i] = 0.0

    # take top-k indices by score
    import numpy as np
    if len(scores) <= k:
        idx = np.argsort(-scores)
    else:
        idx = np.argpartition(-scores, k)[:k]
        idx = idx[np.argsort(-scores[idx])]

    out: list[tuple[dict, float]] = []
    for i in idx:
        if scores[i] <= 0.0:
            break
        out.append((_bm25_state.docs[i], float(scores[i])))
    return out


# ── Reciprocal Rank Fusion ──────────────────────────────────────────
# RRF combines two ranked lists into one by summing 1/(k+rank) over each
# list. it's robust because it ignores the absolute scores (which are
# on different scales for dense vs sparse) and only uses the rank.
# ref: Cormack et al. 2009, k=60 is the canonical default
_RRF_K = 60


def _rrf_fuse(
    dense:  list[tuple[dict, float]],
    sparse: list[tuple[dict, float]],
    top_k:  int,
) -> list[dict]:
    fused: dict[str, dict[str, Any]] = {}

    def _key(doc: dict) -> str:
        # de-dupe by text prefix, same scheme as the dense-only path used.
        return (doc.get("text") or "")[:120]

    for rank, (doc, _score) in enumerate(dense):
        k = _key(doc)
        entry = fused.setdefault(k, {"doc": doc, "score": 0.0, "ranks": [None, None]})
        entry["score"] += 1.0 / (_RRF_K + rank + 1)
        entry["ranks"][0] = rank + 1

    for rank, (doc, _score) in enumerate(sparse):
        k = _key(doc)
        entry = fused.setdefault(k, {"doc": doc, "score": 0.0, "ranks": [None, None]})
        entry["score"] += 1.0 / (_RRF_K + rank + 1)
        entry["ranks"][1] = rank + 1

    ordered = sorted(fused.values(), key=lambda e: -e["score"])[:top_k]
    return [e["doc"] for e in ordered]


# ── Public retrieval ─────────────────────────────────────────────────
async def retrieve(
    query: str,
    top_k: int | None = None,
    lang:  str | None = None,
) -> list[dict[str, Any]]:
    """Hybrid retrieval, dense (Chroma) + sparse (BM25), fused via RRF.

    when lang="cy" we restrict the dense pass to lang=cy passages first,
    then top up. when lang="en" we EXCLUDE lang=cy passages so the
    Welsh-bootstrap docs ("helo (Saesneg: hello)") can't bleed into an
    English reply context. the BM25 pass applies the same lang filter.
    """
    s = get_settings()
    k = top_k or s.rag_top_k

    # Welsh queries get a bigger k, more context = less translation
    # hallucination from the LLM.
    if lang == "cy" and top_k is None:
        k = max(k, 5)
    # English queries also benefit from a slightly larger pool when
    # using hybrid retrieval, the fusion step will narrow back down.
    if lang == "en" and top_k is None:
        k = max(k, 4)

    if not query or not query.strip() or k <= 0:
        return []

    try:
        store = _get_store()
    except Exception as e:
        log.warning("Chroma store init failed: %s", e)
        return []

    # dense pass: pull a wider pool (k * 2) so the fusion step has room
    # to re-rank without losing good candidates that BM25 ranks high.
    dense_pool = max(k * 2, 8)
    dense_results: list[tuple[dict, float]] = []

    def _to_dict(doc, score) -> dict:
        return {
            "text":   doc.page_content,
            "title":  doc.metadata.get("title", doc.metadata.get("source", "corpus")),
            "source": doc.metadata.get("source", ""),
            "lang":   doc.metadata.get("lang", ""),
            "score":  float(score),
        }

    # build the lang filter for Chroma. for cy, only cy. for en, exclude
    # cy (i.e. en-tagged or untagged docs both pass). for None, no filter.
    if lang == "cy":
        dense_filter: dict | None = {"lang": "cy"}
    elif lang == "en":
        # Chroma's where syntax: $ne for not-equal. matches docs where
        # lang is anything other than "cy", including missing/blank.
        # ref: https://docs.trychroma.com/usage-guide#using-where-filters
        dense_filter = {"lang": {"$ne": "cy"}}
    else:
        dense_filter = None

    try:
        kwargs: dict[str, Any] = {"k": dense_pool}
        if dense_filter is not None:
            kwargs["filter"] = dense_filter
        results = store.similarity_search_with_score(query, **kwargs)
        for doc, score in results:
            dense_results.append((_to_dict(doc, score), float(score)))
    except Exception as e:
        log.warning("Chroma dense retrieve failed: %s", e)

    # if the lang-filtered dense pass under-fills, top up with general
    # results. this is a safety net for Welsh queries when the cy slice
    # is sparse on a particular topic.
    if len(dense_results) < k and dense_filter is not None:
        try:
            extra = store.similarity_search_with_score(query, k=dense_pool)
            seen = {d["text"][:120] for d, _ in dense_results}
            for doc, score in extra:
                key = doc.page_content[:120]
                if key in seen:
                    continue
                # even on the topup pass we still respect the language
                # boundary for English queries to keep cy out
                if lang == "en" and doc.metadata.get("lang") == "cy":
                    continue
                dense_results.append((_to_dict(doc, score), float(score)))
                seen.add(key)
        except Exception as e:
            log.debug("Dense topup pass skipped: %s", e)

    # sparse pass: BM25 over the same collection, lang-aware via score
    # zero-ing in the BM25 helper.
    sparse_results = _bm25_search(query, dense_pool, lang)

    # fuse with reciprocal rank fusion. if BM25 isn't available (no
    # rank_bm25 installed) sparse_results is empty and RRF degrades to
    # using only the dense ranking.
    fused = _rrf_fuse(dense_results, sparse_results, k)

    return fused
