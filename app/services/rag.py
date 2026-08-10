"""
hybrid retrieval system combining dense semantic search + sparse keyword matching

chromadb stores 5,500+ indexed passages from multiple sources:
- uwtsd scraped corpus (336 verified pages with source urls)
- curated uwtsd facts (bilingual answers to common factual questions)
- curated knowledge base (intent-driven student responses)
- welsh terminology bootstrap (academic cy vocabulary from BydTermCymru)

retrieval strategy:
1. dense pass: multilingual embeddings (paraphrase-multilingual-MiniLM-L12-v2)
2. sparse pass: bm25 keyword matching via langchain's BM25Retriever, which
   wraps the rank_bm25 implementation of Robertson & Zaragoza's model
3. fusion: langchain's EnsembleRetriever combines both ranked lists using
   reciprocal rank fusion (c=60)
4. language filtering: cy queries prefer welsh passages, en queries exclude cy

this hybrid approach handles both semantic paraphrases ("accommodation options"
-> finds "housing" passages) and exact keywords ("UWTSD fees" -> finds fee pages).

the two retrievers are filtered differently because their backends differ.
chroma supports metadata filtering natively, so the dense pass passes a `where`
clause. bm25 has no such facility, so one index is built per language bucket
and cached.

public api:
  retrieve(query, top_k, lang) - hybrid search with lang-aware filtering
  ingest_documents(docs)       - add new passages to chromadb
  rebuild_bm25()               - refresh keyword index after ingestion
"""
from __future__ import annotations

import logging
import re
from functools import lru_cache
from typing import Any

from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app.config import get_settings


log = logging.getLogger("u-pal-py.rag")


# EnsembleRetriever moved out of the top-level `langchain` namespace in
# LangChain 1.x and now lives in langchain_classic. we import defensively so a
# missing package degrades to dense-only retrieval rather than crashing the
# whole service at import time.
try:
    from langchain_classic.retrievers import EnsembleRetriever
except ImportError:  # pragma: no cover - depends on installed langchain build
    try:
        from langchain.retrievers import EnsembleRetriever  # type: ignore[no-redef]
    except ImportError:
        EnsembleRetriever = None  # type: ignore[assignment,misc]
        log.warning("EnsembleRetriever unavailable, falling back to dense-only")


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
    # the BM25 indexes are now stale. invalidate so the next retrieve()
    # rebuilds them from the latest collection contents.
    _sparse_cache.clear()
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
        _sparse_cache.clear()


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


# ── Tokenisation ─────────────────────────────────────────────────────
# lowercase word tokenisation shared by the BM25 index and query paths so
# the term spaces match. the character class explicitly includes the
# circumflexed vowels used in Welsh orthography (to bach); an ASCII-only
# or naive \w+ pattern would split words such as "ymchwil gwyddonol"
# incorrectly wherever a diacritic appears.
_TOKEN_RE = re.compile(r"[a-zâêîôûŵŷ0-9]+", re.IGNORECASE)


def _tokenise(text: str) -> list[str]:
    if not text:
        return []
    return [t.lower() for t in _TOKEN_RE.findall(text)]


# ── Sparse retrieval (BM25) ──────────────────────────────────────────
# BM25Retriever is LangChain's wrapper around the rank_bm25 package, which
# implements the probabilistic ranking model of Robertson & Zaragoza (2009).
#
# unlike Chroma, BM25 has no metadata filter, so we do the language split by
# building one index per bucket. building an index means tokenising the whole
# collection, so we cache the buckets and only throw them away on ingest.
class _SparseCache:
    def __init__(self) -> None:
        self.by_lang: dict[str | None, Any] = {}
        self.docs: list[Document] | None = None

    def clear(self) -> None:
        self.by_lang = {}
        self.docs = None


_sparse_cache = _SparseCache()


def _collection_documents() -> list[Document]:
    """Read every indexed document out of Chroma once, for BM25 to tokenise.

    _collection.get() returns the whole collection in a single call; for the
    ~5.5k documents in this project that is roughly 60 ms.
    ref: https://docs.trychroma.com/reference/Collection#get
    """
    if _sparse_cache.docs is not None:
        return _sparse_cache.docs
    try:
        store = _get_store()
        raw = store._collection.get(          # type: ignore[attr-defined]
            include=["documents", "metadatas"],
        )
    except Exception as e:
        log.warning("BM25 build failed at fetch step: %s", e)
        _sparse_cache.docs = []
        return []

    texts = raw.get("documents") or []
    metas = raw.get("metadatas") or [{}] * len(texts)
    docs = [
        Document(page_content=t, metadata=dict(m or {}))
        for t, m in zip(texts, metas)
        if t and t.strip()
    ]
    _sparse_cache.docs = docs
    log.info("BM25 source pool loaded: %d docs", len(docs))
    return docs


def _get_sparse_retriever(lang: str | None, k: int) -> Any | None:
    """BM25Retriever over the slice of the collection matching `lang`.

    Returns None when rank_bm25 is unavailable or the bucket is empty, in
    which case retrieve() degrades to a dense-only pass.
    """
    try:
        from langchain_community.retrievers import BM25Retriever
    except ImportError:
        log.warning("BM25Retriever unavailable, hybrid search disabled")
        return None

    retriever = _sparse_cache.by_lang.get(lang)
    if retriever is None:
        docs = _collection_documents()
        # same exclusion the dense filter applies. unlike the BydTermCymru
        # glosses, which BM25 can legitimately match on exact term overlap,
        # a greeting has no terms worth matching.
        docs = [d for d in docs
                if d.metadata.get("title") not in _CONVERSATIONAL_TITLES]
        if lang == "cy":
            docs = [d for d in docs if d.metadata.get("lang") == "cy"]
        elif lang == "en":
            # mirror of the dense filter: exclude cy rather than require en,
            # so untagged documents still participate
            docs = [d for d in docs if d.metadata.get("lang") != "cy"]
        if not docs:
            log.info("BM25 bucket for lang=%s is empty", lang)
            return None
        try:
            retriever = BM25Retriever.from_documents(
                docs, preprocess_func=_tokenise,
            )
        except Exception as e:
            log.warning("BM25 index build failed: %s", e)
            return None
        _sparse_cache.by_lang[lang] = retriever
        log.info("BM25 index built for lang=%s over %d docs", lang, len(docs))

    retriever.k = k
    return retriever


def rebuild_bm25() -> None:
    # external entry point for when callers know the index is stale (e.g.
    # after a CLI ingest). also called automatically by ingest_documents().
    _sparse_cache.clear()
    _collection_documents()


# ── Dense retrieval (Chroma) ─────────────────────────────────────────
# metadata key used to carry the dense distance through EnsembleRetriever,
# which passes Documents along but discards its own fusion scores.
_SCORE_KEY = "_dense_distance"

# must match the source written by scripts/ingest.py for bootstrap terms
_BYDTERMCYMRU_SOURCE = "https://termau.cymru/"

# purely conversational intents from knowledge.json. these carry no
# institutional content ("Shwmae! Croeso i U-Pal"), and routes/chat.py already
# answers them from the intent classifier with retrieval suppressed entirely
# (_CHITCHAT_TAGS, chat.py), so they can never be a useful retrieved passage.
#
# they are excluded here because they were crowding the Welsh ranking. the cy
# slice holds only ~474 content documents against 2666 for English, so the ten
# conversational strings are a far larger share of the candidate pool, and
# being short they sit close to any short query in embedding space - the same
# failure mode as the BydTermCymru glosses above. before this filter they took
# 17 of the 50 dense top-5 slots on the Welsh benchmark queries against 0 on
# the English ones.
#
# matched on title because that is the intent tag as written by
# _load_knowledge(). the tags are lowercase and the curated facts and crawled
# pages use capitalised titles, so there is no collision.
_CONVERSATIONAL_TITLES = ["greeting", "goodbye", "thanks", "how_are_you", "capabilities"]


class _DenseRetriever(BaseRetriever):
    """Chroma similarity search with the bilingual metadata filter applied.

    Wrapped as a BaseRetriever so it can be handed to EnsembleRetriever.
    The store is resolved inside the call rather than held as a field, so the
    lru_cache stays the single source of truth for the connection.
    """

    # size of the candidate pool this retriever contributes to fusion
    k: int = 8
    # the topup pass fires only when the filtered pass returns fewer than the
    # number of passages actually needed downstream, not fewer than the whole
    # pool. topping up against the pool size would drag English passages into
    # every Welsh query, since the cy slice is rarely deep enough to fill it.
    min_results: int = 1
    lang: str | None = None

    def _filter(self) -> dict | None:
        # for cy, only cy. for en, exclude cy (i.e. en-tagged or untagged both
        # pass). for None, no filter.
        # Chroma's where syntax: $ne for not-equal.
        # ref: https://docs.trychroma.com/usage-guide#using-where-filters
        # conversational intents are never a useful retrieval result in either
        # language, so the exclusion is unconditional. see _CONVERSATIONAL_TITLES.
        no_chitchat = {"title": {"$nin": _CONVERSATIONAL_TITLES}}

        if self.lang == "cy":
            # we drop the BydTermCymru glosses from the DENSE pass only.
            # they're 84% of the cy slice (2401 of 2875) and only ~57 chars
            # each, against a corpus chunk mean of 379. a very short string
            # sits close to almost any short query in embedding space, so they
            # were eating 54% of the dense top-5 slots on Welsh queries vs 0%
            # on English. that crowding is what held Welsh dense MRR at 0.20
            # against 0.71 for English, not the encoder itself.
            # BM25 still sees them, which is where a glossary actually earns
            # its keep (exact term overlap).
            return {
                "$and": [
                    {"lang": "cy"},
                    {"source": {"$ne": _BYDTERMCYMRU_SOURCE}},
                    no_chitchat,
                ]
            }
        if self.lang == "en":
            return {"$and": [{"lang": {"$ne": "cy"}}, no_chitchat]}
        return no_chitchat

    def _get_relevant_documents(         # type: ignore[override]
        self, query: str, *, run_manager: Any = None,
    ) -> list[Document]:
        try:
            store = _get_store()
        except Exception as e:
            log.warning("Chroma store init failed: %s", e)
            return []

        dense_filter = self._filter()
        out: list[Document] = []
        seen: set[str] = set()

        try:
            kwargs: dict[str, Any] = {"k": self.k}
            if dense_filter is not None:
                kwargs["filter"] = dense_filter
            for doc, score in store.similarity_search_with_score(query, **kwargs):
                key = doc.page_content[:120]
                if key in seen:
                    continue
                seen.add(key)
                # stash the distance so the caller can apply a relevance
                # threshold after fusion. EnsembleRetriever hands back plain
                # Documents with no scores attached, and this is the last
                # point where we still have the dense distance.
                doc.metadata = {**(doc.metadata or {}), _SCORE_KEY: float(score)}
                out.append(doc)
        except Exception as e:
            log.warning("Chroma dense retrieve failed: %s", e)

        # if the lang-filtered pass under-fills, top up with general results.
        # this is a safety net for Welsh queries when the cy slice is sparse
        # on a particular topic. we still respect the language boundary for
        # English queries so cy passages cannot bleed in.
        if len(out) < self.min_results and dense_filter is not None:
            try:
                for doc, score in store.similarity_search_with_score(query, k=self.k):
                    key = doc.page_content[:120]
                    if key in seen:
                        continue
                    if (doc.metadata or {}).get("lang") == "cy" and self.lang == "en":
                        continue
                    seen.add(key)
                    doc.metadata = {**(doc.metadata or {}), _SCORE_KEY: float(score)}
                    out.append(doc)
            except Exception as e:
                log.debug("Dense topup pass skipped: %s", e)

        return out


# ── Fusion ───────────────────────────────────────────────────────────
# EnsembleRetriever combines the ranked lists using reciprocal rank fusion,
# scoring each document as the sum of 1/(c + rank) over the lists it appears
# in. RRF is used rather than a weighted score combination because dense
# distances and BM25 scores sit on incomparable scales; RRF depends only on
# rank position, so it needs no per-corpus normalisation constants.
# c=60 is the canonical default and matches EnsembleRetriever's own.
_RRF_C = 60

# equal weighting: neither retriever is trusted more a priori. the comparative
# evaluation in Chapter 5 reports dense-only and sparse-only alongside this.
_ENSEMBLE_WEIGHTS = [0.5, 0.5]


def _to_dict(doc: Document) -> dict[str, Any]:
    meta = doc.metadata or {}
    return {
        "text":   doc.page_content,
        "title":  meta.get("title", meta.get("source", "corpus")),
        "source": meta.get("source", ""),
        "lang":   meta.get("lang", ""),
        # cosine distance from the dense pass: 0 is identical, larger is more
        # distant. None when the passage was found by BM25 alone.
        "score":  meta.get(_SCORE_KEY),
    }


# ── Public retrieval ─────────────────────────────────────────────────
async def retrieve(
    query: str,
    top_k: int | None = None,
    lang:  str | None = None,
) -> list[dict[str, Any]]:
    """Hybrid retrieval, dense (Chroma) + sparse (BM25), fused by RRF.

    when lang="cy" we restrict the dense pass to lang=cy passages first,
    then top up. when lang="en" we EXCLUDE lang=cy passages so the
    Welsh-bootstrap docs ("helo (Saesneg: hello)") can't bleed into an
    English reply context. the BM25 pass applies the same lang boundary by
    selecting its index bucket.
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

    # pull a wider pool than we need so the fusion step has room to re-rank
    # without losing good candidates that only one retriever ranks highly.
    pool = max(k * 2, 8)

    dense = _DenseRetriever(k=pool, min_results=k, lang=lang)
    sparse = _get_sparse_retriever(lang, pool)

    # if BM25 or EnsembleRetriever is unavailable the sparse list is empty and
    # we degrade to the dense ranking rather than failing the request.
    if sparse is None or EnsembleRetriever is None:
        try:
            docs = dense.invoke(query)
        except Exception as e:
            log.warning("Dense retrieve failed: %s", e)
            return []
        return [_to_dict(d) for d in docs[:k]]

    try:
        ensemble = EnsembleRetriever(
            retrievers=[dense, sparse],
            weights=_ENSEMBLE_WEIGHTS,
            c=_RRF_C,
        )
        docs = ensemble.invoke(query)
    except Exception as e:
        log.warning("Ensemble retrieve failed, falling back to dense: %s", e)
        try:
            docs = dense.invoke(query)
        except Exception:
            return []

    return [_to_dict(d) for d in docs[:k]]
