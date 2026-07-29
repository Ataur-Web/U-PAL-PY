#!/usr/bin/env python3
"""
build-welsh-map.py — Build welsh-bilingual-map.json from BydTermCymru / TermCymru
=================================================================================

Downloads the full TermCymru dataset (154k Welsh↔English term pairs) from the
HuggingFace datasets server, filters for subjects relevant to a university chatbot,
and merges the results into welsh-bilingual-map.json in the project root.

Source: https://huggingface.co/datasets/TermCymru/TermCymru
Licence: Open Government Licence (OGL) — Crown copyright

Usage:
    python3 scripts/build-welsh-map.py [--all] [--output path/to/map.json]

Flags:
    --all     Include ALL 36 subject areas (not just the curated university ones)
    --output  Output file path (default: welsh-bilingual-map.json in project root)
"""

import sys
import json
import os
import re
import time
import urllib.request
import urllib.parse

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
HF_API        = "https://datasets-server.huggingface.co/rows"
DATASET       = "TermCymru/TermCymru"
BATCH_SIZE    = 100    # rows per API request
MAX_ROWS      = 160000 # upper bound — dataset has ~154k
DELAY         = 0.3    # seconds between requests (be polite)

# Subject areas (Pwnc) relevant to a university chatbot.
# Remove any from this set if you want a narrower map.
RELEVANT_SUBJECTS = {
    "Addysg",              # Education
    "TGCh",               # ICT
    "Iechyd",             # Health
    "Cyllid ac ystadegau", # Finance and Statistics
    "Personél",           # Personnel / HR
    "Cyffredinol",        # General
    "Datblygu economaidd", # Economic Development
    "Gwasanaethau cymdeithasol",  # Social Services
    "Cyfraith",           # Law
    "Llywodraeth leol",   # Local Government
    "Tai",                # Housing
}

# Terms to always skip (overly long or not useful for query augmentation)
MAX_TERM_LEN  = 60
MIN_TERM_LEN  = 2

# Determine project root (one level up from this script)
SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_OUT  = os.path.join(PROJECT_ROOT, "welsh-bilingual-map.json")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def clean(s):
    """Normalise a term to lowercase, strip excess whitespace."""
    if not s or not isinstance(s, str):
        return None
    s = s.strip().lower()
    s = re.sub(r'\s+', ' ', s)
    return s or None


def is_useful(cy, en):
    """Return True if the pair is worth keeping for query augmentation."""
    if not cy or not en:
        return False
    if len(cy) > MAX_TERM_LEN or len(en) > MAX_TERM_LEN:
        return False
    if len(cy) < MIN_TERM_LEN or len(en) < MIN_TERM_LEN:
        return False
    # Skip entries that are just numbers, abbreviations, or reference codes
    if re.match(r'^[\d\s\-/]+$', cy) or re.match(r'^[\d\s\-/]+$', en):
        return False
    # Skip legislation titles
    if re.search(r'\b(act|order|regulation|regulations|cymraeg|saesneg)\b', en.lower()) \
       and len(en.split()) > 4:
        return False
    return True


def fetch_batch(offset, include_all=False):
    """Fetch one batch of rows from the HuggingFace datasets server."""
    params = urllib.parse.urlencode({
        "dataset": DATASET,
        "config":  "default",
        "split":   "train",
        "offset":  offset,
        "length":  BATCH_SIZE,
    })
    url = f"{HF_API}?{params}"
    req = urllib.request.Request(url, headers={"User-Agent": "UPal-Map-Builder/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read().decode("utf-8"))
            rows = data.get("rows", [])
            pairs = []
            for row in rows:
                r_data = row.get("row", {})
                subject = r_data.get("Pwnc", "")
                if not include_all and subject not in RELEVANT_SUBJECTS:
                    continue
                cy  = clean(r_data.get("Cymraeg", ""))
                en  = clean(r_data.get("Saesneg", ""))
                if is_useful(cy, en):
                    pairs.append((cy, en))
            return pairs, len(rows)
    except Exception as e:
        print(f"  WARNING: batch at offset {offset} failed: {e}")
        return [], 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_map(include_all=False, output_path=DEFAULT_OUT):
    print(f"\n{'='*60}")
    print("  BydTermCymru / TermCymru → Welsh Bilingual Map Builder")
    print(f"{'='*60}\n")
    print(f"  Source:   {DATASET} (HuggingFace, OGL licence)")
    print(f"  Subjects: {'ALL' if include_all else ', '.join(sorted(RELEVANT_SUBJECTS))}")
    print(f"  Output:   {output_path}\n")

    # Load existing map so we can merge (hand-curated entries are preserved)
    existing = {}
    if os.path.exists(output_path):
        try:
            with open(output_path, encoding="utf-8") as f:
                existing = json.load(f)
            # Remove metadata comment key
            existing.pop("__comment", None)
            print(f"  Loaded {len(existing)} existing entries to preserve.\n")
        except Exception as e:
            print(f"  WARNING: could not load existing map: {e}\n")

    new_pairs = {}
    offset     = 0
    total_rows = 0
    skipped    = 0

    print("  Downloading from HuggingFace datasets API...")
    while offset < MAX_ROWS:
        pairs, row_count = fetch_batch(offset, include_all)
        if row_count == 0:
            break
        for cy, en in pairs:
            # Don't overwrite hand-curated entries that already exist
            if cy not in existing:
                new_pairs[cy] = en
        total_rows += row_count
        skipped    += (row_count - len(pairs))
        offset     += BATCH_SIZE
        if offset % 5000 == 0:
            print(f"  ... {offset:,} rows processed, {len(new_pairs):,} new pairs found")
        time.sleep(DELAY)

    # Merge: existing (hand-curated) takes priority
    merged = {}
    merged["__comment"] = (
        "Welsh→English term map built from BydTermCymru / TermCymru (gov.wales OGL licence). "
        "Run scripts/build-welsh-map.py to regenerate from the full 154k-entry dataset."
    )
    merged.update(new_pairs)
    merged.update(existing)   # hand-curated entries win

    # Sort keys alphabetically for clean diffs, keeping __comment first
    comment = merged.pop("__comment")
    sorted_merged = {"__comment": comment}
    sorted_merged.update(dict(sorted(merged.items())))

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(sorted_merged, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"  Done!")
    print(f"  Total rows scanned:    {total_rows:,}")
    print(f"  Rows skipped (filter): {skipped:,}")
    print(f"  New terms added:       {len(new_pairs):,}")
    print(f"  Preserved existing:    {len(existing):,}")
    print(f"  Final map size:        {len(sorted_merged) - 1:,} entries")
    print(f"  Output:                {output_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    include_all = "--all" in sys.argv
    try:
        out_idx = sys.argv.index("--output")
        output  = sys.argv[out_idx + 1]
    except (ValueError, IndexError):
        output  = DEFAULT_OUT
    build_map(include_all=include_all, output_path=output)
