#!/usr/bin/env python3
"""
Welsh language detection benchmark - chapter 3.4 / 5.3

Evaluates app.services.welsh.detect_language against requirement F2, which
states the system must identify the language of a query without the user
declaring it.

test set
--------
one hundred queries drawn by stratified random sample from the query phrasings
held in uwtsd-facts.json and knowledge.json, built by
scripts/make_langdetect_sample.py and labelled by hand. neither source file
feeds the detector, whose vocabulary comes from welsh-bilingual-map.json and
BydTermCymru, so the test set is independent of the component under test.
sampling is stratified across three length bands because short queries carry
fewer language signals and are the harder case.

the four prototype task prompts from questionnaire items Q6-Q9 may be added as
a separate stratum once their wording is filled in below; they matter
disproportionately despite being few, because they are inputs real
participants sent through the deployed system.

metrics
-------
overall accuracy, plus per-class recall. accuracy alone is not interpretable
without the class split, since a detector that answered "en" for everything
would still score 50% on a balanced set. Welsh recall is reported separately
because the asymmetry matters: a Welsh query misrouted to English produces an
English answer for a Welsh speaker, which is the failure the system exists to
prevent, whereas the reverse merely produces a Welsh answer for a query that
was already understood.

usage:
    python scripts/benchmark_langdetect.py
"""
from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
for noisy in ("u-pal-py", "app.services.welsh"):
    logging.getLogger(noisy).setLevel(logging.ERROR)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

OUT_DIR = (Path(__file__).resolve().parent.parent
           / "benchmark_results" / "5.3_language_detection")
OUT_DIR.mkdir(parents=True, exist_ok=True)

from app.services import welsh  # noqa: E402

# ── test set ─────────────────────────────────────────────────────────
# the hundred sampled queries are read from the labelled CSV written by
# scripts/make_langdetect_sample.py. "task" items below are the Q6-Q9
# prototype prompts, reported as a separate stratum.

# Q6-Q9 prototype tasks. EDIT THESE to the exact wording used in the
# questionnaire before citing 5.3; the placeholders below are marked so an
# unedited run is obvious in the output rather than silently reported.
TASK_PROMPTS: list[tuple[str, str, str]] = [
    # (label, gold lang, prompt)
    ("Q6 English query",        "en", "<<SET Q6 PROMPT>>"),
    ("Q7 Welsh query",          "cy", "<<SET Q7 PROMPT>>"),
    ("Q8 clarification query",  "en", "<<SET Q8 PROMPT>>"),
    ("Q9 out-of-scope query",   "en", "<<SET Q9 PROMPT>>"),
]


SAMPLE_CSV = OUT_DIR / "langdetect_sample_labelled.csv"


def build_cases() -> list[dict]:
    """The labelled 100-query sample, plus any Q6-Q9 task prompts that have
    been filled in. Strata are reported separately so the headline figure is
    not diluted by items drawn from a different frame."""
    cases: list[dict] = []

    if SAMPLE_CSV.exists():
        with open(SAMPLE_CSV, encoding="utf-8-sig") as f:
            for r in csv.DictReader(f):
                gold = (r.get("Gold language (en/cy/x)") or "").strip().lower()
                if gold not in ("en", "cy"):
                    continue          # "x" = not classifiable, excluded
                cases.append({"source": "sample", "label": r["ID"],
                              "gold": gold, "query": r["Query"]})
    else:
        print(f"!! {SAMPLE_CSV.name} not found; run make_langdetect_sample.py "
              f"and label it first")

    for label, gold, prompt in TASK_PROMPTS:
        if prompt.startswith("<<SET "):
            continue
        cases.append({"source": "task", "label": label, "gold": gold, "query": prompt})
    return cases


def main() -> int:
    welsh.load()
    print("=" * 74)
    print("WELSH LANGUAGE DETECTION BENCHMARK")
    print("=" * 74)
    print(f"vocabulary terms: {welsh.vocab_size()}   "
          f"bilingual pairs: {welsh.bilingual_map_size()}")

    cases = build_cases()
    skipped = sum(1 for _, _, p in TASK_PROMPTS if p.startswith("<<SET "))
    if skipped:
        print(f"\n!! {skipped} of 4 Q6-Q9 task prompts are unset placeholders and were\n"
              f"   EXCLUDED. Results below cover only {len(cases)} items. Fill in\n"
              f"   TASK_PROMPTS before citing this run in 5.3.\n")

    n_cy = sum(1 for c in cases if c["gold"] == "cy")
    n_en = len(cases) - n_cy
    print(f"test set: {len(cases)} queries  ({n_en} English / {n_cy} Welsh)\n")

    rows, errors = [], []
    for c in cases:
        pred = welsh.detect_language(c["query"])
        correct = pred == c["gold"]
        row = {**c, "predicted": pred, "correct": correct}
        rows.append(row)
        if not correct:
            errors.append(row)
        mark = "ok " if correct else "XX "
        tag = f"[{c['source']}]"
        print(f"  {mark}{tag:<10} gold={c['gold']} pred={pred}  {c['query'][:46]}")

    # ── metrics ──────────────────────────────────────────────────────
    total = len(rows)
    n_correct = sum(1 for r in rows if r["correct"])
    acc = n_correct / total

    def recall(lang: str) -> tuple[int, int, float]:
        sub = [r for r in rows if r["gold"] == lang]
        hit = sum(1 for r in sub if r["correct"])
        return hit, len(sub), (hit / len(sub) if sub else 0.0)

    cy_hit, cy_n, cy_rec = recall("cy")
    en_hit, en_n, en_rec = recall("en")

    print()
    print("-" * 74)
    print(f"{'Class':<26}{'Correct':>10}{'Total':>8}{'Recall':>10}")
    print(f"{'Welsh (cy)':<26}{cy_hit:>10}{cy_n:>8}{cy_rec:>10.1%}")
    print(f"{'English (en)':<26}{en_hit:>10}{en_n:>8}{en_rec:>10.1%}")
    print(f"{'Overall accuracy':<26}{n_correct:>10}{total:>8}{acc:>10.1%}")

    # per-stratum, so the 100-query sample can be reported as the headline
    print(f"\n{'Stratum':<26}{'Correct':>10}{'Total':>8}{'Accuracy':>10}")
    for src in ("sample", "task"):
        sub = [r for r in rows if r["source"] == src]
        if not sub:
            continue
        hit = sum(1 for r in sub if r["correct"])
        s_cy = [r for r in sub if r["gold"] == "cy"]
        s_cy_hit = sum(1 for r in s_cy if r["correct"])
        print(f"{src:<26}{hit:>10}{len(sub):>8}{hit / len(sub):>10.1%}"
              f"   (Welsh {s_cy_hit}/{len(s_cy)})")

    print("\nConfusion matrix (rows = gold, cols = predicted)")
    print(f"{'':<10}{'en':>6}{'cy':>6}")
    for gold in ("en", "cy"):
        r_en = sum(1 for r in rows if r["gold"] == gold and r["predicted"] == "en")
        r_cy = sum(1 for r in rows if r["gold"] == gold and r["predicted"] == "cy")
        print(f"{gold:<10}{r_en:>6}{r_cy:>6}")

    if errors:
        print(f"\n{len(errors)} misclassification(s):")
        for e in errors:
            print(f"  gold={e['gold']} pred={e['predicted']}  {e['query']!r}")
    else:
        print("\nno misclassifications")

    # ── export ───────────────────────────────────────────────────────
    per_query = OUT_DIR / "langdetect_per_query.csv"
    with open(per_query, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Source", "Label", "Query", "Gold language",
                    "Predicted language", "Correct"])
        for r in rows:
            w.writerow([r["source"], r["label"], r["query"], r["gold"],
                        r["predicted"], "yes" if r["correct"] else "no"])

    with open(OUT_DIR / "langdetect.json", "w", encoding="utf-8") as f:
        json.dump({
            "n": total, "n_english": en_n, "n_welsh": cy_n,
            "correct": n_correct, "accuracy": acc,
            "welsh_recall": cy_rec, "english_recall": en_rec,
            "task_prompts_unset": skipped,
            "errors": errors, "per_query": rows,
        }, f, ensure_ascii=False, indent=2)

    print(f"\nwrote {per_query.relative_to(OUT_DIR.parent)}")
    print(f"wrote {(OUT_DIR / 'langdetect.json').relative_to(OUT_DIR.parent)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
