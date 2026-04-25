"""Fetch Welsh terminology from BydTermCymru (termau.cymru) and merge into
the local bilingual map used by app/services/welsh.py.

Usage:
    python -m scripts.fetch_termcymru            # fetch + merge, save to app/data/
    python -m scripts.fetch_termcymru --dry-run  # print results, don't save

What it does:
    1. Queries the BydTermCymru public REST API for terms covering the
       subject areas most relevant to a university chatbot.
    2. Normalises the response into {"cy": "...", "en": "..."} pairs.
    3. Merges new pairs into welsh-bilingual-map.json (no duplicates).
    4. Adds Welsh terms to welsh-terms.json (vocab set for detection).

The API:
    GET https://api.termau.cymru/Search/Full/AndOr/0/{query}
    returns JSON, an array of term records each with `termCy` and `termEn` fields.

Run this once to enrich the Welsh corpus.  Re-run whenever you want to
pull fresher data.  Requires internet access.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import httpx

from app.config import DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("fetch-termcymru")

# ── BydTermCymru API ───────────────────────────────────────────────────────────
API_BASE = "https://api.termau.cymru/Search/Full/AndOr/0"
HEADERS  = {
    "Accept":     "application/json",
    "User-Agent": "U-PAL chatbot / Welsh term enrichment (educational)",
}

# Subject-area search terms most relevant to a university chatbot.
# These return domain-specific bilingual pairs far beyond the general vocab.
SEARCH_QUERIES = [
    # University life
    "university", "prifysgol", "student", "myfyriwr",
    "degree", "gradd", "lecture", "darlith",
    "module", "modiwl", "semester", "tymor",
    "tuition", "dysgu", "assessment", "asesiad",
    "dissertation", "traethawd", "campus", "campws",
    "accommodation", "llety", "scholarship", "ysgoloriaeth",
    "enrolment", "cofrestru", "graduation", "graddio",
    # Administration
    "application", "cais", "admission", "mynediad",
    "fees", "ffioedd", "finance", "cyllid",
    "bursary", "bwrsari", "loan", "benthyciad",
    # Wellbeing
    "wellbeing", "llesiant", "support", "cymorth",
    "counselling", "cwnsela", "disability", "anabledd",
    # Welsh language / bilingualism
    "Welsh language", "Cymraeg", "bilingual", "dwyieithog",
    "translation", "cyfieithiad",
    # Technology (for IT questions)
    "computer", "cyfrifiadur", "software", "meddalwedd",
    "internet", "rhyngrwyd", "password", "cyfrinair",
    "email", "e-bost",
    # Programmes
    "engineering", "peirianneg", "science", "gwyddoniaeth",
    "medicine", "meddygaeth", "law", "y gyfraith",
    "business", "busnes", "art", "celf",
    "education", "addysg", "psychology", "seicoleg",
]

# de-duplicate, the same word may appear in different cases across queries
SEARCH_QUERIES = list(dict.fromkeys(q.lower() for q in SEARCH_QUERIES))


def _fetch_terms(query: str, client: httpx.Client) -> list[dict]:
    """Return list of raw term records for *query* from BydTermCymru."""
    url = f"{API_BASE}/{httpx.URL(query)}"
    try:
        r = client.get(f"{API_BASE}/{query}", headers=HEADERS, timeout=15)
        r.raise_for_status()
        data = r.json()
        # API may return a top-level list or a wrapper dict
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            # Try common wrapper keys
            for key in ("terms", "results", "data", "items"):
                if isinstance(data.get(key), list):
                    return data[key]
        return []
    except httpx.HTTPStatusError as e:
        log.warning("HTTP %s for query %r, skipping", e.response.status_code, query)
        return []
    except Exception as e:
        log.warning("Error fetching %r: %s", query, e)
        return []


def _normalise(record: dict) -> tuple[str, str] | None:
    """Extract (cy, en) from a BydTermCymru record.

    The API has used several different key names over time; we try them all.
    """
    cy = (
        record.get("termCy")
        or record.get("welsh")
        or record.get("cy")
        or record.get("WelshTerm")
        or ""
    ).strip()
    en = (
        record.get("termEn")
        or record.get("english")
        or record.get("en")
        or record.get("EnglishTerm")
        or ""
    ).strip()
    if cy and en and cy.lower() != en.lower():
        return cy.lower(), en.lower()
    return None


def fetch_all() -> dict[str, str]:
    """Fetch all queries and return a {cy: en} dict."""
    pairs: dict[str, str] = {}
    with httpx.Client(follow_redirects=True) as client:
        for i, query in enumerate(SEARCH_QUERIES):
            log.info("[%d/%d] Querying: %s", i + 1, len(SEARCH_QUERIES), query)
            records = _fetch_terms(query, client)
            count = 0
            for rec in records:
                result = _normalise(rec)
                if result:
                    cy, en = result
                    pairs.setdefault(cy, en)
                    count += 1
            log.info("  → %d usable pairs (running total: %d)", count, len(pairs))
            # be polite, cap at 2 requests per second
            if i < len(SEARCH_QUERIES) - 1:
                time.sleep(0.5)
    return pairs


def merge_bilingual_map(new_pairs: dict[str, str]) -> tuple[int, int]:
    """Merge *new_pairs* into welsh-bilingual-map.json.

    Returns (pairs_added, total_pairs).
    """
    path = DATA_DIR / "welsh-bilingual-map.json"
    existing: list[dict] = []
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            existing = []

    # Build a set of existing (cy) keys for deduplication
    existing_cy = {
        (p.get("cy") or p.get("welsh") or "").strip().lower()
        for p in existing
        if isinstance(p, dict)
    }

    added = 0
    for cy, en in new_pairs.items():
        if cy not in existing_cy:
            existing.append({"cy": cy, "en": en})
            existing_cy.add(cy)
            added += 1

    path.write_text(
        json.dumps(existing, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return added, len(existing)


def merge_vocab(new_pairs: dict[str, str]) -> tuple[int, int]:
    """Add Welsh sides of *new_pairs* into welsh-terms.json.

    Returns (terms_added, total_terms).
    """
    path = DATA_DIR / "welsh-terms.json"
    existing: list = []
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            existing = raw if isinstance(raw, list) else raw.get("terms", [])
        except Exception:
            existing = []

    existing_set = {t.strip().lower() for t in existing if isinstance(t, str)}
    added = 0
    for cy in new_pairs:
        if cy not in existing_set:
            existing.append(cy)
            existing_set.add(cy)
            added += 1

    path.write_text(
        json.dumps(existing, ensure_ascii=False),
        encoding="utf-8",
    )
    return added, len(existing)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch BydTermCymru Welsh terms")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print fetched pairs without saving")
    args = parser.parse_args()

    log.info("Fetching Welsh terms from BydTermCymru ...")
    log.info("API: %s", API_BASE)
    log.info("Queries: %d", len(SEARCH_QUERIES))

    pairs = fetch_all()

    if not pairs:
        log.error(
            "No pairs fetched!  The API endpoint may have changed.\n"
            "Check https://termau.cymru for the current API docs.\n"
            "Current endpoint tried: %s/{query}", API_BASE
        )
        return

    log.info("Fetched %d unique Welsh-English pairs", len(pairs))

    if args.dry_run:
        log.info("--- DRY RUN, sample output (first 20) ---")
        for cy, en in list(pairs.items())[:20]:
            print(f"  {cy}  ->  {en}")
        return

    added_bi, total_bi = merge_bilingual_map(pairs)
    log.info("Bilingual map: +%d new pairs  (total: %d)", added_bi, total_bi)

    added_vocab, total_vocab = merge_vocab(pairs)
    log.info("Welsh vocab:   +%d new terms  (total: %d)", added_vocab, total_vocab)

    log.info("")
    log.info("Done.  Restart the backend (start-everything.bat) to pick up changes.")
    log.info("The health endpoint should now show higher bilingualTerms and welshVocab counts.")


if __name__ == "__main__":
    main()
