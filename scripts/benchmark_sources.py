#!/usr/bin/env python3
"""
Requirement F8 benchmark - source attribution

F8 states: "The system must display the knowledge base sources used to generate
each response." That is three separate claims, and this benchmark tests each
one, deliberately separating what can be verified mechanically from what needs
human judgement.

  1. COVERAGE   - are sources displayed at all, and how many?
  2. INTEGRITY  - does every displayed source correspond to a document that is
                  actually in the knowledge base? this is the claim that
                  matters most, because 4.5 rejected LLM-generated citations
                  precisely on the grounds that they can be fabricated. if a
                  displayed URL is absent from the index, the attribution is
                  invented and the design decision has failed.
  3. VALIDITY   - does each displayed URL resolve to a live institutional page?
                  a citation a student cannot follow does not let them verify
                  anything.

relevance - whether a source actually addresses the query - is NOT measured
here. it is a judgement, not a fact about the system, and is rated separately
by a human on the sample this script writes to sources_relevance_TO_RATE.csv.
mixing a rated quantity into the mechanical results would obscure which
findings are objective.

the same 100 queries used for the F2 evaluation are reused, so the two
requirements are assessed over one consistent workload.

usage:
    python scripts/benchmark_sources.py [--base-url http://localhost:3001]
                                        [--no-url-check]
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

import httpx

logging.basicConfig(level=logging.WARNING)
for noisy in ("httpx", "chromadb", "sentence_transformers", "u-pal-py"):
    logging.getLogger(noisy).setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
OUT_DIR = ROOT / "benchmark_results" / "5.4_source_attribution"
OUT_DIR.mkdir(parents=True, exist_ok=True)
# the query set is shared with the 5.3 evaluation, so both requirements are
# assessed over one consistent workload
SAMPLE = (ROOT / "benchmark_results" / "5.3_language_detection"
          / "langdetect_sample_labelled.csv")

PAUSE = 0.6          # between chat requests
URL_PAUSE = 1.0      # between institutional URL checks, matching the crawler's
                     # one-second politeness delay in 4.3
RELEVANCE_SAMPLE = 20   # queries drawn for human relevance rating


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:3001")
    ap.add_argument("--no-url-check", action="store_true",
                    help="skip live HTTP checks against uwtsd.ac.uk")
    args = ap.parse_args()

    if not SAMPLE.exists():
        print(f"missing {SAMPLE.name}")
        return 1
    with open(SAMPLE, encoding="utf-8-sig") as f:
        cases = [r for r in csv.DictReader(f)
                 if (r.get("Gold language (en/cy/x)") or "").strip().lower() in ("en", "cy")]

    print("=" * 78)
    print("REQUIREMENT F8 - SOURCE ATTRIBUTION")
    print("=" * 78)

    # ── the set of source URLs actually present in the index ─────────
    from app.services import rag
    store = rag._get_store()
    raw = store._collection.get(include=["metadatas"])
    kb_sources = {(m or {}).get("source") for m in raw["metadatas"]}
    kb_sources.discard(None)
    print(f"knowledge base holds {len(kb_sources)} distinct source values")

    rows: list[dict] = []          # one row per displayed source
    per_query: list[dict] = []     # one row per query

    with httpx.Client(base_url=args.base_url, timeout=120.0) as client:
        health = client.get("/api/health").json()
        provider = health.get("provider", "unknown")
        print(f"provider: {provider}   queries: {len(cases)}\n")

        for c in cases:
            gold = c["Gold language (en/cy/x)"].strip().lower()
            q = c["Query"]
            r = client.post("/api/chat", json={"message": q})
            body = r.json() if r.status_code == 200 else {}
            srcs = body.get("sources", []) or []
            per_query.append({
                "id": c["ID"], "query": q, "lang": gold,
                "reply_lang": body.get("lang", ""),
                "intent": body.get("intent") or "",
                "n_sources": len(srcs), "status": r.status_code,
            })
            for i, s in enumerate(srcs, start=1):
                url = (s.get("url") or "").strip()
                host = urlparse(url).netloc.lower() if url else ""
                rows.append({
                    "id": c["ID"], "query": q, "query_lang": gold,
                    "rank": i,
                    "title": s.get("title", ""),
                    "category": s.get("category", ""),
                    "source_lang": s.get("language", ""),
                    "url": url,
                    "in_knowledge_base": url in kb_sources,
                    "is_uwtsd_domain": host.endswith("uwtsd.ac.uk"),
                    "excerpt": (s.get("excerpt", "") or "").replace("\n", " ")[:160],
                    "http_status": "",
                })
            print(f"  [{gold}] {len(srcs)} source(s)  {q[:52]}")
            time.sleep(PAUSE)

    # ── live URL validation, deduplicated and rate-limited ───────────
    if not args.no_url_check:
        uniq = sorted({r["url"] for r in rows if r["url"]})
        print(f"\nchecking {len(uniq)} distinct URLs "
              f"({URL_PAUSE}s apart, matching the crawler delay)...")
        status: dict[str, str] = {}
        with httpx.Client(timeout=20.0, follow_redirects=True,
                          headers={"User-Agent": "U-Pal-eval/1.0 (dissertation evaluation)"}) as c2:
            for u in uniq:
                try:
                    resp = c2.head(u)
                    if resp.status_code >= 400:      # some servers refuse HEAD
                        resp = c2.get(u)
                    status[u] = str(resp.status_code)
                except Exception as e:
                    status[u] = f"error: {e.__class__.__name__}"
                time.sleep(URL_PAUSE)
        for r in rows:
            r["http_status"] = status.get(r["url"], "")

    # ── metrics ──────────────────────────────────────────────────────
    n_q = len(per_query)
    with_src = [p for p in per_query if p["n_sources"] > 0]
    n_src = len(rows)

    def pct(a, b):
        return f"{a}/{b} ({a / b:.0%})" if b else "-"

    print("\n" + "-" * 78)
    print("COVERAGE")
    print(f"  queries displaying at least one source : {pct(len(with_src), n_q)}")
    print(f"  total sources displayed                : {n_src}")
    print(f"  mean sources per query                 : {n_src / n_q:.2f}")
    dist = Counter(p["n_sources"] for p in per_query)
    print("  sources per query: " +
          ", ".join(f"{k}->{dist[k]}" for k in sorted(dist)))
    for lang, label in (("en", "English"), ("cy", "Welsh")):
        sub = [p for p in per_query if p["lang"] == lang]
        s_n = sum(p["n_sources"] for p in sub)
        got = [p for p in sub if p["n_sources"] > 0]
        print(f"  {label:<8} queries with a source: {pct(len(got), len(sub))}"
              f"   mean {s_n / len(sub):.2f}")

    print("\nINTEGRITY  (is each displayed source really in the knowledge base?)")
    in_kb = sum(1 for r in rows if r["in_knowledge_base"])
    print(f"  displayed sources traced to an indexed document : {pct(in_kb, n_src)}")
    bad = [r for r in rows if not r["in_knowledge_base"]]
    for r in bad[:10]:
        print(f"    NOT IN INDEX: {r['url']!r} (query {r['id']})")

    print("\nVALIDITY")
    uw = sum(1 for r in rows if r["is_uwtsd_domain"])
    print(f"  sources on a uwtsd.ac.uk domain : {pct(uw, n_src)}")
    if not args.no_url_check:
        ok = sum(1 for r in rows if r["http_status"] == "200")
        print(f"  sources whose URL returned HTTP 200 : {pct(ok, n_src)}")
        badu = Counter(r["http_status"] for r in rows if r["http_status"] != "200")
        for k, v in badu.most_common():
            print(f"    {k}: {v}")

    print("\nLANGUAGE CONSISTENCY")
    match = sum(1 for r in rows if r["source_lang"] == r["query_lang"])
    print(f"  sources tagged in the query's language : {pct(match, n_src)}")

    # ── exports ──────────────────────────────────────────────────────
    src_csv = OUT_DIR / "sources_per_source.csv"
    with open(src_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Query ID", "Query", "Query language", "Source rank",
                    "Title", "Category", "Source language", "URL",
                    "In knowledge base", "UWTSD domain", "HTTP status", "Excerpt"])
        for r in rows:
            w.writerow([r["id"], r["query"], r["query_lang"], r["rank"], r["title"],
                        r["category"], r["source_lang"], r["url"],
                        "yes" if r["in_knowledge_base"] else "NO",
                        "yes" if r["is_uwtsd_domain"] else "no",
                        r["http_status"], r["excerpt"]])

    q_csv = OUT_DIR / "sources_per_query.csv"
    with open(q_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Query ID", "Query", "Query language", "Intent",
                    "Sources displayed", "HTTP status"])
        for p in per_query:
            w.writerow([p["id"], p["query"], p["lang"], p["intent"],
                        p["n_sources"], p["status"]])

    # human relevance rating sheet, drawn from queries that showed sources
    rate = [r for r in rows if r["id"] in
            {p["id"] for p in with_src[:RELEVANCE_SAMPLE]}]
    rate_csv = OUT_DIR / "sources_relevance_TO_RATE.csv"
    with open(rate_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["Query ID", "Query", "Source rank", "Title", "URL", "Excerpt",
                    "Relevant? (2=directly answers, 1=related, 0=unrelated)"])
        for r in rate:
            w.writerow([r["id"], r["query"], r["rank"], r["title"], r["url"],
                        r["excerpt"], ""])

    with open(OUT_DIR / "sources.json", "w", encoding="utf-8") as f:
        json.dump({"provider": provider, "n_queries": n_q,
                   "n_sources": n_src, "per_query": per_query,
                   "per_source": rows}, f, ensure_ascii=False, indent=2)

    print(f"\nwrote {src_csv.relative_to(ROOT)}")
    print(f"wrote {q_csv.relative_to(ROOT)}")
    print(f"wrote {rate_csv.relative_to(ROOT)}  <- rate these by hand")
    return 0


if __name__ == "__main__":
    sys.exit(main())
