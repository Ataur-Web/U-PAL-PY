#!/usr/bin/env python3
"""
end-to-end response latency benchmark - chapter 3.4 / 5.2

submits thirty chat requests to a running U-Pal backend (fifteen English,
fifteen Welsh) and reports mean, median, 95th percentile, minimum and maximum
wall-clock latency, overall and per language, against requirement N1
(mean < 8 s).

the fifteen queries per language pair up: each row below carries one English
and one Welsh phrasing of the same information need, mirroring the parallel
design of the retrieval benchmark so the language comparison is like for like.
latency is measured around the whole HTTP round trip - language detection,
retrieval, generation and serialisation - because N1 is a requirement on what
a student experiences, not on any single stage.

each request is sent sequentially with no concurrency, matching how a single
student interacts with the chat interface. a small pause between requests
avoids provider-side rate limiting contaminating the measurement.

usage:
    python scripts/benchmark_latency.py [--base-url http://localhost:3001]
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import httpx

OUT_DIR = (Path(__file__).resolve().parent.parent
           / "benchmark_results" / "5.2_latency")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# (english, welsh) pairs. drawn from the same information needs as the
# retrieval benchmark plus five conversational/support queries, so the set
# covers the mix of traffic the chatbot actually receives.
QUERY_PAIRS: list[tuple[str, str]] = [
    ("Where is the SA1 campus?",
     "Ble mae campws SA1?"),
    ("How much are the tuition fees?",
     "Faint yw'r ffioedd dysgu?"),
    ("How do I apply for accommodation?",
     "Sut ydw i'n gwneud cais am lety?"),
    ("What are the library opening hours?",
     "Beth yw oriau agor y llyfrgell?"),
    ("How do I apply to UWTSD?",
     "Sut mae gwneud cais i PCYDDS?"),
    ("How do I apply for student finance?",
     "Sut mae gwneud cais am gyllid myfyrwyr?"),
    ("When is graduation?",
     "Pryd mae'r seremoni raddio?"),
    ("How do I travel between campuses?",
     "Sut allaf deithio rhwng campysau?"),
    ("What wellbeing support is available?",
     "Pa gefnogaeth lles sydd ar gael?"),
    ("What courses does the university offer?",
     "Pa gyrsiau mae'r brifysgol yn eu cynnig?"),
    ("How do I contact my personal tutor?",
     "Sut ydw i'n cysylltu â'm tiwtor personol?"),
    ("What can I do at the students' union?",
     "Beth alla i ei wneud yn undeb y myfyrwyr?"),
    ("How do I get IT support?",
     "Sut ydw i'n cael cymorth TG?"),
    ("What financial hardship support exists?",
     "Pa gymorth caledi ariannol sydd ar gael?"),
    ("How do I report extenuating circumstances?",
     "Sut ydw i'n rhoi gwybod am amgylchiadau esgusodol?"),
]

# pause between requests, seconds. long enough to stay clear of provider
# rate limits, short enough that the run stays under five minutes.
PAUSE = 1.0


def summarise(samples: list[float]) -> dict[str, float]:
    xs = sorted(samples)
    n = len(xs)
    # nearest-rank p95, the convention most latency reporting uses
    p95 = xs[max(0, min(n - 1, round(0.95 * n) - 1))]
    return {
        "n":      n,
        "mean":   statistics.mean(xs),
        "median": statistics.median(xs),
        "p95":    p95,
        "min":    xs[0],
        "max":    xs[-1],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:3001")
    ap.add_argument("--tag", default="",
                    help="suffix for output filenames, e.g. --tag ollama "
                         "writes latency_ollama_per_query.csv")
    args = ap.parse_args()
    tag = f"_{args.tag}" if args.tag else ""

    print("=" * 70)
    print("RESPONSE LATENCY BENCHMARK")
    print("=" * 70)

    with httpx.Client(base_url=args.base_url, timeout=90.0) as client:
        # confirm the backend is up and record which provider answers, since
        # 5.2 must name the provider the figures describe
        health = client.get("/api/health").json()
        provider = health.get("llm_provider") or health.get("provider") or "unknown"
        model = health.get("model") or health.get("llm_model") or ""
        print(f"backend: {args.base_url}   provider: {provider} {model}")

        # one untimed warmup per language so first-request costs (index
        # loading, connection setup) are excluded, matching the retrieval
        # benchmark's treatment of build cost
        print("warming up (untimed)...", flush=True)
        for msg, lang in (("hello", "en"), ("helo", "cy")):
            client.post("/api/chat", json={"message": msg, "runningLang": lang})

        rows: list[dict] = []
        for q_en, q_cy in QUERY_PAIRS:
            for lang, query in (("en", q_en), ("cy", q_cy)):
                t0 = time.perf_counter()
                r = client.post("/api/chat",
                                json={"message": query, "runningLang": lang})
                elapsed = time.perf_counter() - t0
                ok = r.status_code == 200
                body = r.json() if ok else {}
                rows.append({
                    "language":   lang,
                    "query":      query,
                    "latency_s":  round(elapsed, 3),
                    "status":     r.status_code,
                    "reply_lang": body.get("lang", ""),
                    "n_sources":  len(body.get("sources", [])),
                    "reply_chars": len(body.get("reply", "")),
                })
                flag = "" if ok else f"  <== HTTP {r.status_code}"
                print(f"  [{lang}] {elapsed:6.2f}s  {query[:52]:<54}{flag}")
                time.sleep(PAUSE)

    good = [r for r in rows if r["status"] == 200]
    failed = len(rows) - len(good)
    if failed:
        print(f"\nWARNING: {failed} request(s) failed; failed requests are "
              f"excluded from the latency statistics but recorded in the CSV")

    subsets = {
        "All":     [r["latency_s"] for r in good],
        "English": [r["latency_s"] for r in good if r["language"] == "en"],
        "Welsh":   [r["latency_s"] for r in good if r["language"] == "cy"],
    }

    print()
    print(f"{'Subset':<10}{'n':>4}{'Mean':>8}{'Median':>8}{'p95':>8}{'Min':>8}{'Max':>8}")
    print("-" * 54)
    summary = {}
    for label, xs in subsets.items():
        if not xs:
            continue
        s = summarise(xs)
        summary[label] = s
        print(f"{label:<10}{s['n']:>4}{s['mean']:>8.2f}{s['median']:>8.2f}"
              f"{s['p95']:>8.2f}{s['min']:>8.2f}{s['max']:>8.2f}")

    if "All" in summary:
        verdict = "MET" if summary["All"]["mean"] < 8.0 else "NOT MET"
        print(f"\nN1 (mean < 8 s): {verdict}  (mean {summary['All']['mean']:.2f}s)")

    per_query = OUT_DIR / f"latency{tag}_per_query.csv"
    with open(per_query, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    with open(OUT_DIR / f"latency{tag}.json", "w", encoding="utf-8") as f:
        json.dump({
            "provider": provider,
            "model": model,
            "n_requests": len(rows),
            "n_failed": failed,
            "summary": summary,
            "per_query": rows,
        }, f, ensure_ascii=False, indent=2)

    print(f"\nwrote {per_query.relative_to(OUT_DIR.parent)}")
    print(f"wrote {(OUT_DIR / f'latency{tag}.json').relative_to(OUT_DIR.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
