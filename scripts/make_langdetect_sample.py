#!/usr/bin/env python3
"""
Build the language-detection test sample - chapter 3.4 / 5.3

Draws a stratified random sample of 100 queries from the query phrasings
already held in the knowledge base, and writes them to a CSV with an EMPTY
ground-truth column for a human to complete.

why the labels are left blank
-----------------------------
the obvious way to label these automatically would be to look for Welsh
function words and diacritics - which is exactly what the detector under test
does. labelling that way would measure the detector against a copy of its own
heuristics and could not fail. ground truth is therefore established by human
annotation, and this script deliberately writes no suggested label and sorts
the sample randomly rather than by language, so the annotator reads each item
rather than confirming a guess.

sampling frame
--------------
the "questions" lists in uwtsd-facts.json and the "patterns" lists in
knowledge.json. these are phrasings written during development to represent
how students ask each question, in both languages. neither file feeds the
detector, whose vocabulary comes from welsh-bilingual-map.json and
BydTermCymru, so the sample is independent of the component being tested.

the thirty queries used for the latency benchmark are excluded, so the two
evaluations do not share items.

stratification
--------------
a language guess (heuristic, used ONLY to balance the draw, never written to
the file) crossed with three length bands, because query length drives the
detector's behaviour: short queries carry fewer signals and are the harder
case. sampling proportionally within bands stops the set collapsing into
two-word fragments, which would be unrepresentative of real student queries
while also being artificially difficult.

usage:
    python scripts/make_langdetect_sample.py [--n 100] [--seed 42]
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA = ROOT / "app" / "data"
OUT_DIR = ROOT / "benchmark_results" / "5.3_language_detection"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# heuristic ONLY for balancing the draw. never written to the output file.
_CY_HINT = re.compile(
    r"[âêîôûŵŷ]|\b(ble|beth|sut|pryd|faint|mae|ydw|ydy|yw|pa|oes|yn|i'?n|"
    r"gwneud|cais|myfyrwyr|prifysgol|cymraeg|campws|llyfrgell|ffioedd|"
    r"cymorth|gallaf|allaf|angen|eisiau|rhaid|hoffwn)\b",
    re.IGNORECASE,
)
_TOKENS = re.compile(r"[A-Za-zâêîôûŵŷ0-9']+")

# the latency benchmark queries, excluded so the two test sets stay disjoint
LATENCY_QUERIES = {
    "where is the sa1 campus?", "ble mae campws sa1?",
    "how much are the tuition fees?", "faint yw'r ffioedd dysgu?",
    "how do i apply for accommodation?", "sut ydw i'n gwneud cais am lety?",
    "what are the library opening hours?", "beth yw oriau agor y llyfrgell?",
    "how do i apply to uwtsd?", "sut mae gwneud cais i pcydds?",
    "how do i apply for student finance?", "sut mae gwneud cais am gyllid myfyrwyr?",
    "when is graduation?", "pryd mae'r seremoni raddio?",
    "how do i travel between campuses?", "sut allaf deithio rhwng campysau?",
    "what wellbeing support is available?", "pa gefnogaeth lles sydd ar gael?",
    "what courses does the university offer?", "pa gyrsiau mae'r brifysgol yn eu cynnig?",
    "how do i contact my personal tutor?", "sut ydw i'n cysylltu â'm tiwtor personol?",
    "what can i do at the students' union?", "beth alla i ei wneud yn undeb y myfyrwyr?",
    "how do i get it support?", "sut ydw i'n cael cymorth tg?",
    "what financial hardship support exists?", "pa gymorth caledi ariannol sydd ar gael?",
    "how do i report extenuating circumstances?",
    "sut ydw i'n rhoi gwybod am amgylchiadau esgusodol?",
}


def band(n_tokens: int) -> str:
    if n_tokens <= 3:
        return "short (1-3 tokens)"
    if n_tokens <= 7:
        return "medium (4-7 tokens)"
    return "long (8+ tokens)"


def collect() -> list[dict]:
    pool: dict[str, dict] = {}

    facts = json.loads((DATA / "uwtsd-facts.json").read_text(encoding="utf-8"))
    for e in facts if isinstance(facts, list) else []:
        for q in e.get("questions") or []:
            if isinstance(q, str) and q.strip():
                pool.setdefault(q.strip().lower(),
                                {"query": q.strip(), "source": "uwtsd-facts.json"})

    kn = json.loads((DATA / "knowledge.json").read_text(encoding="utf-8"))
    ents = kn.get("intents") if isinstance(kn, dict) else kn
    for e in ents or []:
        pats = e.get("patterns")
        items = pats if isinstance(pats, list) else []
        if isinstance(pats, dict):
            items = [x for v in pats.values() for x in v]
        for q in items:
            if isinstance(q, str) and q.strip():
                pool.setdefault(q.strip().lower(),
                                {"query": q.strip(), "source": "knowledge.json"})

    rows = []
    for key, v in pool.items():
        if key in LATENCY_QUERIES:
            continue
        toks = _TOKENS.findall(v["query"])
        if not toks:
            continue
        rows.append({
            **v,
            "tokens": len(toks),
            "band": band(len(toks)),
            "_guess": "cy" if _CY_HINT.search(v["query"]) else "en",
        })
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    rows = collect()
    print(f"sampling frame: {len(rows)} unique queries "
          f"(latency-benchmark items excluded)")

    per_lang = args.n // 2
    bands = ["short (1-3 tokens)", "medium (4-7 tokens)", "long (8+ tokens)"]
    # even split across bands, remainder to the medium band which is the
    # most representative of real student queries
    quota = {b: per_lang // 3 for b in bands}
    quota[bands[1]] += per_lang - sum(quota.values())

    picked: list[dict] = []
    for guess in ("en", "cy"):
        for b in bands:
            cands = [r for r in rows if r["_guess"] == guess and r["band"] == b]
            want = quota[b]
            if len(cands) < want:
                print(f"  note: only {len(cands)} candidates for {guess}/{b}, "
                      f"wanted {want}; shortfall redistributed")
                take = cands
            else:
                take = rng.sample(cands, want)
            picked += take
        # top up this language from any band if a shortfall occurred
        have = sum(1 for p in picked if p["_guess"] == guess)
        if have < per_lang:
            rest = [r for r in rows
                    if r["_guess"] == guess and r not in picked]
            picked += rng.sample(rest, min(per_lang - have, len(rest)))

    rng.shuffle(picked)   # random order, so the annotator cannot pattern-match

    out = OUT_DIR / "langdetect_sample_TO_LABEL.csv"
    with open(out, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["ID", "Query", "Tokens", "Length band", "Source file",
                    "Gold language (en/cy/x)"])
        for i, r in enumerate(picked, start=1):
            w.writerow([i, r["query"], r["tokens"], r["band"], r["source"], ""])

    n_short = sum(1 for p in picked if p["band"].startswith("short"))
    n_med = sum(1 for p in picked if p["band"].startswith("medium"))
    n_long = sum(1 for p in picked if p["band"].startswith("long"))
    print(f"\nwrote {out.relative_to(ROOT)}  ({len(picked)} queries)")
    print(f"  length bands: {n_short} short / {n_med} medium / {n_long} long")
    print("\nFill the last column with en, cy, or x (x = not classifiable, e.g.")
    print("a bare proper noun that belongs to no language). Leave nothing blank.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
