"""Download a curated subset of Google's Natural Questions and inject into Chroma.

Natural Questions is a corpus of real Google search queries paired with
human-annotated answers from Wikipedia. The dataset is published by
Google Research and freely available on Hugging Face.
ref: https://ai.google.com/research/NaturalQuestions
ref: https://huggingface.co/datasets/google-research-datasets/nq_open

Why this script exists:
We noticed in testing that U-Pal would deflect general academic questions
(things like "what is photosynthesis?" or "explain calculus") to Student
Services, even though it should be able to chat about any topic a curious
student might bring up. The OpenOrca pull covers a lot of instructional
Q&A but skews towards "explain like I'm five" replies. Natural Questions
brings real, encyclopedic factual answers into the index so the bot can
answer general-knowledge questions without hallucinating.

What this script does:
  1. streams the nq_open variant of Natural Questions from Hugging Face
     (the simplified short-answer subset, much smaller than the full corpus).
  2. filters for academic-leaning questions using the same keyword set we
     already use for OpenOrca, plus an extra "general knowledge" set.
  3. formats each as a Q&A pair.
  4. saves to app/data/naturalquestions-qa.json.
  5. ingests into Chroma so the RAG layer can retrieve them.

Usage:
    python -m scripts.fetch_naturalquestions
    python -m scripts.fetch_naturalquestions --sample 10000
    python -m scripts.fetch_naturalquestions --dry-run

Note: the nq_open subset is roughly 100 MB on first download. Subsequent
runs are served from the local Hugging Face cache.
"""
from __future__ import annotations

import argparse
import json
import logging
import re

from app.config import DATA_DIR


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("fetch-natural-questions")

OUTPUT_FILE = DATA_DIR / "naturalquestions-qa.json"


# we keep the keyword set wider than fetch_openorca on purpose. the point
# of pulling Natural Questions is to broaden coverage, not to narrow it.
# anything a sixth-form / undergrad student is likely to type counts.
_KEYWORDS = re.compile(
    r"\b("
    # academic subjects
    r"math|maths|algebra|calculus|geometry|trigonometry|statistic|"
    r"physics|chemistry|biology|science|astronomy|geology|"
    r"history|geography|politic|economic|philosophy|psychology|sociolog|"
    r"english|literature|grammar|language|linguistic|"
    r"computer|computing|programming|software|algorithm|data|"
    r"engineering|architect|design|art|music|theatre|theater|film|"
    r"law|medicine|nursing|health|nutrition|"
    # study-skills + university words (overlap with openorca on purpose)
    r"universit|college|student|tuition|lecture|professor|"
    r"semest|degree|bachelor|master|phd|dissertation|thesis|"
    r"assignment|essay|coursework|deadline|exam|assessment|"
    r"scholarship|grant|loan|financial aid|bursary|fee|"
    # generic learning prompts
    r"explain|how does|what is|why is|define|describe|when did|who was|"
    r"who is|where is|meaning of|difference between|how to|"
    r"compare|list|summari[sz]e"
    r")\b",
    re.IGNORECASE,
)


def _is_relevant(question: str, answer: str) -> bool:
    combined = f"{question} {answer}"
    return bool(_KEYWORDS.search(combined))


def _is_quality(question: str, answer: str) -> bool:
    # very short answers (one or two words) are usually too sparse to be
    # useful in the RAG index. very long ones blow the context window.
    if not answer or not question:
        return False
    if len(answer) < 12 or len(answer) > 800:
        return False
    if len(question) < 8 or len(question) > 400:
        return False
    return True


def _normalise_answer(raw: object) -> str:
    # nq_open stores the answer as a list[str] of accepted answer aliases.
    # we join them with " / " so the LLM sees all variants.
    if isinstance(raw, list):
        clean = [str(a).strip() for a in raw if str(a).strip()]
        return " / ".join(clean)
    return str(raw).strip()


def fetch_and_filter(sample_size: int) -> list[dict]:
    try:
        from datasets import load_dataset  # type: ignore[import]
    except ImportError:
        log.error(
            "The 'datasets' package is not installed.\n"
            "Run: .venv\\Scripts\\python.exe -m pip install datasets"
        )
        raise SystemExit(1)

    log.info("Loading Natural Questions (nq_open), streaming first %d rows ...", sample_size)
    log.info("First run downloads roughly 100 MB to the HF cache.")

    # ref: https://huggingface.co/datasets/google-research-datasets/nq_open
    ds = load_dataset(
        "google-research-datasets/nq_open",
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
        if seen % 5_000 == 0:
            log.info("  scanned %d rows -> %d matches so far", seen, len(matches))

        question = (row.get("question") or "").strip()
        answer = _normalise_answer(row.get("answer"))

        if _is_relevant(question, answer) and _is_quality(question, answer):
            matches.append({
                "question": question,
                "response": answer,
                "source": "natural_questions",
            })

    log.info("Finished scanning %d rows -> %d relevant Q&A pairs", seen, len(matches))
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
            metadata={"title": q[:80], "source": "natural_questions"},
        ))

    if not docs:
        log.warning("No documents to ingest.")
        return

    added = rag.ingest_documents(docs)
    log.info("Ingested %d Natural Questions documents into Chroma.", added)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch Google Natural Questions general-knowledge Q&A into Chroma",
    )
    parser.add_argument("--sample", type=int, default=30_000,
                        help="How many nq_open rows to scan (default: 30000)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print matches without saving or ingesting")
    args = parser.parse_args()

    qa_pairs = fetch_and_filter(args.sample)

    if not qa_pairs:
        log.warning("No matches found. Try increasing --sample or adjusting keywords.")
        return

    if args.dry_run:
        log.info("--- DRY RUN, first 10 matches ---")
        for i, p in enumerate(qa_pairs[:10]):
            print(f"\n[{i+1}] Q: {p['question'][:120]}")
            print(f"     A: {p['response'][:200]}")
        log.info("(Total matches: %d, not saved)", len(qa_pairs))
        return

    save_qa(qa_pairs)
    ingest_to_chroma(qa_pairs)

    log.info("")
    log.info("Done. Restart start-everything.bat to pick up the enriched Chroma index.")


if __name__ == "__main__":
    main()
