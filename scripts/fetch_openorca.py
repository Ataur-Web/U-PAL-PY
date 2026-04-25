"""Download a curated subset of OpenOrca and inject it into Chroma.

OpenOrca (https://huggingface.co/datasets/Open-Orca/OpenOrca) is a 4.3M-row
dataset of high-quality instruction-following conversations generated from
GPT-4 and GPT-3.5.  For a UWTSD student chatbot we only want:
  - Education / university / student-life topics
  - High-quality responses (GPT-4 labelled rows)
  - Short-to-medium length (under 800 chars), easier for the LLM to retrieve

What this script does:
  1. Streams a sample of OpenOrca via Hugging Face datasets (no full download).
  2. Filters rows matching university/education keywords.
  3. Formats each as a Q&A pair.
  4. Saves to app/data/openorca-qa.json.
  5. Ingests into Chroma so future RAG queries can benefit from them.

Usage:
    python -m scripts.fetch_openorca               # default: sample 50k rows
    python -m scripts.fetch_openorca --sample 20000
    python -m scripts.fetch_openorca --dry-run     # print matches, don't save

Requirements:
    pip install datasets   (already in requirements.txt after this PR)

NOTE: First run streams ~500 MB from Hugging Face.  Subsequent runs use
      the local cache (~/.cache/huggingface/datasets/).
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

from app.config import DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("fetch-openorca")

OUTPUT_FILE = DATA_DIR / "openorca-qa.json"

# Keywords that suggest the Q&A is relevant to a student / university context.
# We match against the combined question + response text (lowercased).
_KEYWORDS = re.compile(
    r"\b("
    r"universit|college|student|tuition|lecture|professor|professor|"
    r"semest|degree|bachelor|master|phd|dissertation|thesis|"
    r"assignment|essay|coursework|deadline|exam|assessment|"
    r"accommodation|campus|library|module|curriculum|enrollment|"
    r"scholarship|grant|loan|financial aid|bursary|fees|"
    r"wellbeing|mental health|stress|anxiety|support service|"
    r"career|internship|placement|graduate|alumni|"
    r"application|ucas|personal statement|interview|offer|"
    r"study skills|revision|research|academic|"
    r"explain|how does|what is|why is|define|describe|"
    r"help me understand|step.by.step|summarise|summarize"
    r")\b",
    re.IGNORECASE,
)

# Prefer GPT-4 generated rows (higher quality)
_PREFERRED_IDS = {"niv.246340", "niv."}  # OpenOrca uses `id` field


def _is_relevant(row: dict) -> bool:
    q = row.get("question", "")
    r = row.get("response", "")
    sys = row.get("system_prompt", "")
    combined = f"{sys} {q} {r}"
    return bool(_KEYWORDS.search(combined))


def _is_quality(row: dict) -> bool:
    """Prefer GPT-4 labelled rows and short-to-medium responses."""
    r = row.get("response", "")
    # Exclude very short (unhelpful) or very long (hard to index) responses
    if len(r) < 80 or len(r) > 1200:
        return False
    return True


def fetch_and_filter(sample_size: int) -> list[dict]:
    """Stream OpenOrca and return filtered relevant rows."""
    try:
        from datasets import load_dataset   # type: ignore[import]
    except ImportError:
        log.error(
            "The 'datasets' package is not installed.\n"
            "Run: .venv\\Scripts\\python.exe -m pip install datasets"
        )
        raise SystemExit(1)

    log.info("Loading OpenOrca dataset (streaming first %d rows) …", sample_size)
    log.info("First run will download ~500 MB to the HF cache.")

    ds = load_dataset(
        "Open-Orca/OpenOrca",
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    matches: list[dict] = []
    seen: int = 0

    for row in ds:
        if seen >= sample_size:
            break
        seen += 1
        if seen % 10_000 == 0:
            log.info("  scanned %d rows → %d matches so far", seen, len(matches))

        if _is_relevant(row) and _is_quality(row):
            matches.append({
                "question": row.get("question", "").strip(),
                "response": row.get("response", "").strip(),
                "source":   "openorca",
            })

    log.info("Finished scanning %d rows → %d relevant Q&A pairs", seen, len(matches))
    return matches


def save_qa(qa_pairs: list[dict]) -> None:
    existing: list[dict] = []
    if OUTPUT_FILE.exists():
        try:
            existing = json.loads(OUTPUT_FILE.read_text(encoding="utf-8"))
        except Exception:
            existing = []

    existing_qs = {p["question"] for p in existing}
    added = [p for p in qa_pairs if p["question"] not in existing_qs]
    existing.extend(added)

    OUTPUT_FILE.write_text(
        json.dumps(existing, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log.info("Saved %d new pairs to %s  (total: %d)", len(added), OUTPUT_FILE, len(existing))


def ingest_to_chroma(qa_pairs: list[dict]) -> None:
    from langchain_core.documents import Document
    from app.services import rag

    docs: list[Document] = []
    for pair in qa_pairs:
        q = pair["question"]
        r = pair["response"]
        docs.append(Document(
            page_content=f"Q: {q}\n\nA: {r}",
            metadata={"title": q[:80], "source": "openorca"},
        ))

    if not docs:
        log.warning("No documents to ingest.")
        return

    added = rag.ingest_documents(docs)
    log.info("Ingested %d OpenOrca documents into Chroma.", added)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch OpenOrca education Q&A into Chroma")
    parser.add_argument("--sample",  type=int, default=50_000,
                        help="How many OpenOrca rows to scan (default: 50000)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print matches without saving or ingesting")
    args = parser.parse_args()

    qa_pairs = fetch_and_filter(args.sample)

    if not qa_pairs:
        log.warning("No matches found.  Try increasing --sample or adjusting keywords.")
        return

    if args.dry_run:
        log.info("--- DRY RUN, first 10 matches ---")
        for i, p in enumerate(qa_pairs[:10]):
            print(f"\n[{i+1}] Q: {p['question'][:120]}")
            print(f"     A: {p['response'][:200]}...")
        log.info("(Total matches: %d, not saved)", len(qa_pairs))
        return

    save_qa(qa_pairs)
    ingest_to_chroma(qa_pairs)

    log.info("")
    log.info("Done.  Restart start-everything.bat to pick up the enriched Chroma index.")


if __name__ == "__main__":
    main()
