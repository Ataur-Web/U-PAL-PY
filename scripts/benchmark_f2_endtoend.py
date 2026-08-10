#!/usr/bin/env python3
"""
Requirement F2 end-to-end benchmark - chapter 3.4 / 5.3

F2 states: "The system must detect the user input language without the user
needing to declare it." That is a claim about the deployed system, not about
one function, so this benchmark exercises the whole request path.

what is measured
----------------
each of the 100 labelled queries is POSTed to /api/chat carrying ONLY the
message field. runningLang is deliberately omitted, because supplying it would
declare the language and defeat the requirement being tested. two outcomes are
then recorded per query:

  1. detected language  - the "lang" field the API returns, i.e. what the
     system decided the query was.
  2. reply language     - the language the answer is actually written in,
     classified from the reply text by Lingua, an off-the-shelf identifier
     that shares no code or data with U-Pal.

the second is the one that matters to a student. a system could detect Welsh
correctly and still answer in English, and only measuring the reply text
catches that. reporting both separates a detection failure from a generation
failure.

ground truth comes from benchmark_results/langdetect_sample_labelled.csv.

usage:
    python scripts/benchmark_f2_endtoend.py [--base-url http://localhost:3001]
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

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "benchmark_results" / "5.3_language_detection"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SAMPLE = OUT_DIR / "langdetect_sample_labelled.csv"

PAUSE = 0.6   # seconds between requests, to stay clear of provider rate limits

try:
    from lingua import Language, LanguageDetectorBuilder
    _LINGUA = (LanguageDetectorBuilder
               .from_languages(Language.WELSH, Language.ENGLISH)
               .with_preloaded_language_models()
               .build())
except Exception:                                     # pragma: no cover
    _LINGUA = None


def reply_language(text: str) -> str:
    """Classify the answer text independently of U-Pal's own detector."""
    if not _LINGUA or not text or not text.strip():
        return "?"
    v = _LINGUA.detect_language_of(text)
    return {"WELSH": "cy", "ENGLISH": "en"}.get(v.name if v else "", "?")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:3001")
    args = ap.parse_args()

    if not SAMPLE.exists():
        print(f"missing {SAMPLE.name}; run make_langdetect_sample.py and label it")
        return 1
    if _LINGUA is None:
        print("!! lingua not installed - reply-language column will be '?'.\n"
              "   pip install lingua-language-detector")

    with open(SAMPLE, encoding="utf-8-sig") as f:
        cases = [r for r in csv.DictReader(f)
                 if (r.get("Gold language (en/cy/x)") or "").strip().lower() in ("en", "cy")]

    print("=" * 78)
    print("REQUIREMENT F2 - END-TO-END LANGUAGE HANDLING")
    print("=" * 78)

    rows: list[dict] = []
    with httpx.Client(base_url=args.base_url, timeout=120.0) as client:
        health = client.get("/api/health").json()
        provider = health.get("provider", "unknown")
        model = health.get("anthropicModel") if provider == "anthropic" else health.get("ollamaModel")
        print(f"provider: {provider} ({model})   queries: {len(cases)}")
        print("runningLang is NOT sent, so the system must detect the language itself\n")

        for c in cases:
            gold = c["Gold language (en/cy/x)"].strip().lower()
            q = c["Query"]
            t0 = time.perf_counter()
            # ONLY the message: no language hint of any kind
            r = client.post("/api/chat", json={"message": q})
            elapsed = time.perf_counter() - t0
            ok = r.status_code == 200
            body = r.json() if ok else {}
            detected = body.get("lang", "")
            reply = body.get("reply", "")
            rlang = reply_language(reply)

            row = {
                "id": c["ID"], "query": q, "gold": gold,
                "detected": detected, "reply_lang": rlang,
                "detect_ok": detected == gold,
                "reply_ok": rlang == gold,
                "status": r.status_code,
                "intent": body.get("intent") or "",
                "n_sources": len(body.get("sources", [])),
                "latency_s": round(elapsed, 2),
                "reply": reply.replace("\n", " ")[:400],
            }
            rows.append(row)
            d = "ok" if row["detect_ok"] else "XX"
            g = "ok" if row["reply_ok"] else "XX"
            print(f"  detect:{d} reply:{g}  gold={gold} det={detected or '-'} "
                  f"rep={rlang}  {q[:44]}")
            time.sleep(PAUSE)

    good = [r for r in rows if r["status"] == 200]
    failed = len(rows) - len(good)

    def rate(pred_key: str, subset: list[dict]) -> tuple[int, int, float]:
        hit = sum(1 for r in subset if r[pred_key])
        n = len(subset)
        return hit, n, (hit / n if n else 0.0)

    cy = [r for r in good if r["gold"] == "cy"]
    en = [r for r in good if r["gold"] == "en"]

    print("\n" + "-" * 78)
    print(f"{'Measure':<34}{'Welsh':>14}{'English':>14}{'Overall':>14}")
    for key, label in (("detect_ok", "Language detected correctly"),
                       ("reply_ok", "Answer written in that language")):
        c_h, c_n, c_r = rate(key, cy)
        e_h, e_n, e_r = rate(key, en)
        a_h, a_n, a_r = rate(key, good)
        print(f"{label:<34}{f'{c_h}/{c_n} ({c_r:.0%})':>14}"
              f"{f'{e_h}/{e_n} ({e_r:.0%})':>14}{f'{a_h}/{a_n} ({a_r:.0%})':>14}")

    if failed:
        print(f"\n{failed} request(s) did not return HTTP 200")

    det_err = [r for r in good if not r["detect_ok"]]
    rep_err = [r for r in good if not r["reply_ok"]]
    print(f"\ndetection errors: {len(det_err)}")
    for r in det_err:
        print(f"   gold={r['gold']} detected={r['detected']}  {r['query']!r}")
    print(f"answer-language errors: {len(rep_err)}")
    for r in rep_err:
        print(f"   gold={r['gold']} detected={r['detected']} reply={r['reply_lang']}"
              f"  {r['query']!r}")

    lat = [r["latency_s"] for r in good]
    if lat:
        print(f"\nmean latency {statistics.mean(lat):.2f}s over {len(lat)} requests")

    out = OUT_DIR / "f2_endtoend_per_query.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["ID", "Query", "Gold language", "Detected language",
                    "Answer language", "Detection correct", "Answer language correct",
                    "Intent", "Sources shown", "Latency (s)", "HTTP status",
                    "Reply (truncated)"])
        for r in rows:
            w.writerow([r["id"], r["query"], r["gold"], r["detected"], r["reply_lang"],
                        "yes" if r["detect_ok"] else "no",
                        "yes" if r["reply_ok"] else "no",
                        r["intent"], r["n_sources"], r["latency_s"], r["status"],
                        r["reply"]])

    with open(OUT_DIR / "f2_endtoend.json", "w", encoding="utf-8") as f:
        json.dump({"provider": provider, "model": model, "n": len(rows),
                   "n_failed": failed, "per_query": rows}, f,
                  ensure_ascii=False, indent=2)

    print(f"\nwrote {out.relative_to(ROOT)}")
    print(f"wrote {(OUT_DIR / 'f2_endtoend.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
