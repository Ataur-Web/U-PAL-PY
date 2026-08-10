#!/usr/bin/env python3
"""
retrieval configuration benchmark - dense vs sparse vs hybrid

compares the three retrieval configurations described in chapter 3.4 using
top-k hit rate (k = 1, 3, 5) and mean reciprocal rank.

experimental design
-------------------
twenty queries, ten english and ten welsh. the two halves are PARALLEL: each
of ten curated facts contributes one english query and one welsh query
expressing the same information need. because the underlying fact is indexed
once per language, the english and welsh conditions differ only in language,
not in question difficulty, so any gap between them is attributable to the
system rather than to the queries.

ground truth
------------
each query targets exactly one indexed document, identified by the pair
(title, language). the curated facts are ingested one document per language
and are not chunked, so one fact in one language is exactly one indexed
chunk - matching the "single ground-truth document chunk" definition in 3.4.
matching on language as well as title is deliberate: a welsh query that
retrieves the english version of the right fact is NOT counted as a hit,
because it would not give a welsh speaker a welsh answer.

configurations
--------------
dense-only   ChromaDB similarity search with the bilingual metadata filter
sparse-only  BM25Retriever (rank_bm25) over the matching language bucket
hybrid       EnsembleRetriever fusing both with reciprocal rank fusion, c=60

each configuration returns its top 5 results, drawn through the same pipeline
it would use in deployment. retrieval is exercised directly rather than over
http, so no llm is involved and the measurement isolates retrieval.

usage:
    python scripts/benchmark_retrieval.py
"""
from __future__ import annotations

import asyncio
import csv
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
for noisy in ("httpx", "chromadb", "sentence_transformers", "u-pal-py"):
    logging.getLogger(noisy).setLevel(logging.ERROR)

# allow running as a plain script as well as `python -m scripts.benchmark_retrieval`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# everything this script produces lands here, so the evaluation chapter has one
# folder to cite and the repo root stays clean
OUT_DIR = (Path(__file__).resolve().parent.parent
           / "benchmark_results" / "5.1_retrieval")
OUT_DIR.mkdir(parents=True, exist_ok=True)

from app.services import rag  # noqa: E402

# how many results each configuration returns
TOP_K = 5

# (fact id, english query, welsh query)
# fact ids resolve to index titles via the same transform scripts/ingest.py
# applies, so ground truth is derived rather than transcribed by hand.
QUERY_SET: list[tuple[str, str, str]] = [
    ("campus-sa1-location",
     "Where is the SA1 campus?",
     "ble mae campws SA1"),
    ("campus-carmarthen-location",
     "Where is the Carmarthen campus?",
     "ble mae campws Caerfyrddin"),
    ("fees-general",
     "How much are the tuition fees?",
     "faint yw'r ffioedd dysgu?"),
    ("fees-international-undergraduate",
     "How much are the fees for international students?",
     "faint yw ffioedd myfyrwyr rhyngwladol?"),
    ("accommodation-apply",
     "How do I apply for accommodation?",
     "sut ydw i'n gwneud cais am lety?"),
    ("library-hours",
     "What are the library opening hours?",
     "oriau agor y llyfrgell"),
    ("how-to-apply",
     "How do I apply to UWTSD?",
     "sut mae gwneud cais i PCYDDS?"),
    ("student-finance",
     "How do I apply for student finance?",
     "sut mae gwneud cais am gyllid myfyrwyr?"),
    ("inter-campus-travel",
     "How do I travel between campuses?",
     "sut allaf deithio rhwng campysau?"),
    ("graduation",
     "When is graduation?",
     "pryd mae'r seremoni raddio?"),
]


def title_for(fact_id: str) -> str:
    """Mirror the id -> title transform used by scripts.ingest._load_facts."""
    return " ".join(
        w.upper() if len(w) <= 3 and any(c.isdigit() for c in w) else w.capitalize()
        for w in fact_id.replace("-", " ").replace("_", " ").split()
    )


def rank_of(results: list[dict], gold_title: str, gold_lang: str) -> int | None:
    """STRICT: 1-based rank of the canonical fact document, else None.

    Requires the exact curated fact entry, in the query's own language.
    """
    for i, r in enumerate(results, start=1):
        if r.get("title") == gold_title and r.get("lang") == gold_lang:
            return i
    return None


def relaxed_gold_sources(fact_id: str) -> set[str]:
    """RELAXED: the institutional page the fact cites, if any.

    A student asking "how much are the tuition fees" is served equally well by
    the curated fact and by the UWTSD fee schedule page the fact was mapped to.
    Treating only the fact as correct understates the system, so the relaxed
    condition also accepts any chunk drawn from that page. The mapping comes
    from scripts/map_source_urls.py and is validated against the crawl, so the
    accepted set is derived rather than chosen to flatter the result.
    """
    facts = json.loads(
        (Path(__file__).resolve().parent.parent / "app" / "data"
         / "uwtsd-facts.json").read_text(encoding="utf-8")
    )
    for e in facts:
        if e.get("id") == fact_id:
            url = e.get("source_url")
            return {url} if url else set()
    return set()


def rank_of_relaxed(
    results: list[dict], gold_title: str, gold_lang: str, gold_sources: set[str],
) -> int | None:
    """1-based rank of the first acceptable document under the relaxed rule."""
    for i, r in enumerate(results, start=1):
        if r.get("title") == gold_title and r.get("lang") == gold_lang:
            return i
        if r.get("source") in gold_sources:
            return i
    return None


# ── the three configurations ─────────────────────────────────────────
def run_dense(query: str, lang: str) -> list[dict]:
    retriever = rag._DenseRetriever(k=TOP_K, min_results=TOP_K, lang=lang)
    return [rag._to_dict(d) for d in retriever.invoke(query)[:TOP_K]]


def run_sparse(query: str, lang: str) -> list[dict]:
    retriever = rag._get_sparse_retriever(lang, TOP_K)
    if retriever is None:
        return []
    return [rag._to_dict(d) for d in retriever.invoke(query)[:TOP_K]]


async def run_hybrid(query: str, lang: str) -> list[dict]:
    # the production path, which widens the pool then fuses with RRF
    return await rag.retrieve(query, top_k=TOP_K, lang=lang)


CONFIGS = ["Dense-only (ChromaDB)", "Sparse-only (BM25)", "Hybrid (EnsembleRetriever, RRF)"]


async def main() -> int:
    print("=" * 78)
    print("RETRIEVAL CONFIGURATION BENCHMARK")
    print("=" * 78)
    print(f"queries: {len(QUERY_SET) * 2}  ({len(QUERY_SET)} English / {len(QUERY_SET)} Welsh)")
    print(f"top-k returned per configuration: {TOP_K}")
    print()

    # warm the store and both BM25 buckets so build cost is excluded
    print("warming index...", flush=True)
    await rag.retrieve("warmup", top_k=TOP_K, lang="en")
    await rag.retrieve("cynhesu", top_k=TOP_K, lang="cy")
    print("ready\n", flush=True)

    rows: list[dict] = []
    for fact_id, q_en, q_cy in QUERY_SET:
        gold = title_for(fact_id)
        gold_src = relaxed_gold_sources(fact_id)
        for lang, query in (("en", q_en), ("cy", q_cy)):
            results = {
                CONFIGS[0]: run_dense(query, lang),
                CONFIGS[1]: run_sparse(query, lang),
                CONFIGS[2]: await run_hybrid(query, lang),
            }
            row = {
                "fact_id": fact_id,
                "gold_title": gold,
                "language": lang,
                "query": query,
            }
            for c in CONFIGS:
                row[f"strict::{c}"] = rank_of(results[c], gold, lang)
                row[f"relaxed::{c}"] = rank_of_relaxed(results[c], gold, lang, gold_src)
            rows.append(row)

            marks = "  ".join(
                f"{c.split()[0][:6]}:{row[f'strict::{c}'] or '-'}"
                f"/{row[f'relaxed::{c}'] or '-'}" for c in CONFIGS
            )
            print(f"  [{lang}] {query[:40]:<42} {marks}")

    # ── metrics ──────────────────────────────────────────────────────
    def metrics(subset: list[dict], config: str, mode: str) -> dict:
        n = len(subset)
        rs = [r[f"{mode}::{config}"] for r in subset]
        hit = lambda k: sum(1 for x in rs if x is not None and x <= k) / n
        mrr = sum((1.0 / x) for x in rs if x is not None) / n
        return {"Hit@1": hit(1), "Hit@3": hit(3), "Hit@5": hit(5), "MRR": mrr}

    subsets = {
        "All (n=20)": rows,
        "English (n=10)": [r for r in rows if r["language"] == "en"],
        "Welsh (n=10)": [r for r in rows if r["language"] == "cy"],
    }

    print()
    for mode, caption in (("strict", "STRICT - canonical fact entry only"),
                          ("relaxed", "RELAXED - fact entry OR the page it cites")):
        print("#" * 78)
        print(f"# {caption}")
        print("#" * 78)
        for label, subset in subsets.items():
            print(f"\n{label}")
            print("-" * 78)
            print(f"{'Configuration':<34}{'Hit@1':>10}{'Hit@3':>10}{'Hit@5':>10}{'MRR':>10}")
            for c in CONFIGS:
                m = metrics(subset, c, mode)
                print(f"{c:<34}{m['Hit@1']:>10.2f}{m['Hit@3']:>10.2f}"
                      f"{m['Hit@5']:>10.2f}{m['MRR']:>10.2f}")
        print()

    # ── export ───────────────────────────────────────────────────────
    per_query = OUT_DIR / "retrieval_per_query.csv"
    with open(per_query, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Fact ID", "Ground-truth title", "Language", "Query",
                    "Strict rank (dense)", "Strict rank (sparse)", "Strict rank (hybrid)",
                    "Relaxed rank (dense)", "Relaxed rank (sparse)", "Relaxed rank (hybrid)"])
        for r in rows:
            w.writerow([
                r["fact_id"], r["gold_title"], r["language"], r["query"],
                *[r[f"strict::{c}"] or "not retrieved" for c in CONFIGS],
                *[r[f"relaxed::{c}"] or "not retrieved" for c in CONFIGS],
            ])

    summary = OUT_DIR / "retrieval_summary.csv"
    with open(summary, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Relevance rule", "Subset", "Configuration",
                    "Hit@1", "Hit@3", "Hit@5", "MRR"])
        for mode in ("strict", "relaxed"):
            for label, subset in subsets.items():
                for c in CONFIGS:
                    m = metrics(subset, c, mode)
                    w.writerow([mode, label, c, f"{m['Hit@1']:.2f}",
                                f"{m['Hit@3']:.2f}", f"{m['Hit@5']:.2f}",
                                f"{m['MRR']:.2f}"])

    with open(OUT_DIR / "retrieval.json", "w", encoding="utf-8") as f:
        json.dump({
            "top_k": TOP_K,
            "n_queries": len(rows),
            "summary": {
                mode: {
                    label: {c: metrics(subset, c, mode) for c in CONFIGS}
                    for label, subset in subsets.items()
                }
                for mode in ("strict", "relaxed")
            },
            "per_query": rows,
        }, f, ensure_ascii=False, indent=2)

    print(f"wrote {per_query.relative_to(OUT_DIR.parent)}")
    print(f"wrote {summary.relative_to(OUT_DIR.parent)}")
    print(f"wrote {(OUT_DIR / 'retrieval.json').relative_to(OUT_DIR.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
