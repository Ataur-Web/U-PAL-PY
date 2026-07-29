"""Download a curated subset of locailabs/nemotron-chat-welsh into Chroma.

The Welsh response quality in U-Pal was being criticised as inaccurate
or unnatural ("Pwy sy'n gweithio ar y cwrs?" instead of a natural reply
to "Dwi angen help gyda fy nghwrs"). The root cause is that our Welsh
coverage in the retrieval index is mostly the curated UWTSD JSON corpus
plus the BydTermCymru bilingual map, both of which are short. The LLM
has nothing conversational to ground its Welsh phrasing in.

This script pulls a curated subset of the Nemotron Chat Welsh dataset
(a Welsh-translated chat corpus published by locailabs on Hugging Face)
into Chroma so the retrieval layer can return natural Welsh phrasing
when a Welsh query lands.
ref: https://huggingface.co/datasets/locailabs/nemotron-chat-welsh

Usage:
    python -m scripts.fetch_welsh_chat
    python -m scripts.fetch_welsh_chat --sample 5000
    python -m scripts.fetch_welsh_chat --dry-run

Note: first run downloads roughly 200 MB to the local Hugging Face cache.
"""
from __future__ import annotations

import argparse
import json
import logging
import re

from app.config import DATA_DIR


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("fetch-welsh-chat")

OUTPUT_FILE = DATA_DIR / "welsh-chat-qa.json"


# Welsh diacritics, function words, and high-frequency vocabulary.
# we use them to verify a row is actually Welsh and not English that
# slipped through the translation. ref: https://termau.cymru
_WELSH_DIACRITICS = re.compile(r"[âêîôûŵŷÂÊÎÔÛŴŶ]")
_WELSH_FUNCTION_WORDS = {
    "yr", "y", "yn", "ac", "a", "ar", "am", "at", "i", "o", "mewn", "gan",
    "mae", "ydy", "yw", "oedd", "fydd", "bydd", "cael", "wedi", "wrth",
    "chi", "ti", "fi", "ni", "nhw", "ef", "hi", "fo",
    "dw", "dwi", "rwy", "rydw", "rydych", "rydyn", "rwyt",
    "beth", "pam", "pryd", "ble", "pwy", "sut", "faint",
    "shwmae", "helo", "diolch", "croeso", "iawn", "ddim",
    "prifysgol", "myfyriwr", "cymraeg", "cymru", "cwrs", "ffioedd",
}


def _is_welsh(text: str) -> bool:
    # multi-signal Welsh detector. one diacritic OR two function-word hits
    # is enough to call a string Welsh. mirrors app/services/welsh.py.
    if not text:
        return False
    if _WELSH_DIACRITICS.search(text):
        return True
    tokens = set(re.findall(r"[a-zâêîôûŵŷ']+", text.lower()))
    return len(tokens & _WELSH_FUNCTION_WORDS) >= 2


def _is_quality(question: str, answer: str) -> bool:
    # too short = unhelpful, too long = blows context window
    if not question or not answer:
        return False
    if len(question) < 8 or len(question) > 500:
        return False
    if len(answer) < 30 or len(answer) > 1500:
        return False
    return True


def _extract_pair(row: dict) -> tuple[str, str] | None:
    """Try several common chat schemas and return (question, answer)."""
    # schema 1, OpenAI-style messages
    msgs = row.get("messages")
    if isinstance(msgs, list) and msgs:
        user_msg = next((m for m in msgs if m.get("role") == "user"), None)
        asst_msg = next((m for m in msgs if m.get("role") == "assistant"), None)
        if user_msg and asst_msg:
            return (
                str(user_msg.get("content", "")).strip(),
                str(asst_msg.get("content", "")).strip(),
            )

    # schema 2, ShareGPT-style conversations
    convs = row.get("conversations")
    if isinstance(convs, list) and convs:
        user_turn = next((c for c in convs if c.get("from") in ("human", "user")), None)
        asst_turn = next((c for c in convs if c.get("from") in ("gpt", "assistant")), None)
        if user_turn and asst_turn:
            return (
                str(user_turn.get("value", "")).strip(),
                str(asst_turn.get("value", "")).strip(),
            )

    # schema 3, flat prompt/response (Nemotron variants often use this)
    for q_key, a_key in (
        ("prompt", "response"),
        ("question", "answer"),
        ("input", "output"),
        ("instruction", "output"),
    ):
        if row.get(q_key) and row.get(a_key):
            return (str(row[q_key]).strip(), str(row[a_key]).strip())

    return None


def fetch_and_filter(sample_size: int) -> list[dict]:
    try:
        from datasets import load_dataset  # type: ignore[import]
    except ImportError:
        log.error(
            "The 'datasets' package is not installed.\n"
            "Run: .venv\\Scripts\\python.exe -m pip install datasets"
        )
        raise SystemExit(1)

    log.info("Loading nemotron-chat-welsh, streaming first %d rows ...", sample_size)
    log.info("First run downloads roughly 200 MB to the HF cache.")

    # ref: https://huggingface.co/datasets/locailabs/nemotron-chat-welsh
    ds = load_dataset(
        "locailabs/nemotron-chat-welsh",
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
        if seen % 2_000 == 0:
            log.info("  scanned %d rows -> %d Welsh pairs so far", seen, len(matches))

        pair = _extract_pair(row)
        if not pair:
            continue
        question, answer = pair

        if not _is_quality(question, answer):
            continue

        # both sides should be Welsh, the answer especially. if the user
        # turn is in English but the assistant replies in Welsh that's
        # still useful (covers EN->CY style transfer).
        if not _is_welsh(answer):
            continue

        matches.append({
            "question": question,
            "response": answer,
            "source": "nemotron_chat_welsh",
        })

    log.info("Finished scanning %d rows -> %d Welsh Q&A pairs", seen, len(matches))
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
        # tagging the document with lang=cy lets the retriever (or any
        # future filter) prefer Welsh passages when the query is Welsh.
        docs.append(Document(
            page_content=f"Q: {q}\n\nA: {r}",
            metadata={"title": q[:80], "source": "nemotron_chat_welsh", "lang": "cy"},
        ))

    if not docs:
        log.warning("No documents to ingest.")
        return

    added = rag.ingest_documents(docs)
    log.info("Ingested %d Welsh-chat documents into Chroma.", added)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch nemotron-chat-welsh into Chroma to improve Welsh response quality",
    )
    parser.add_argument("--sample", type=int, default=15_000,
                        help="How many rows to scan (default: 15000)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print matches without saving or ingesting")
    args = parser.parse_args()

    qa_pairs = fetch_and_filter(args.sample)

    if not qa_pairs:
        log.warning("No matches found. Try increasing --sample or check the dataset schema.")
        return

    if args.dry_run:
        log.info("--- DRY RUN, first 5 matches ---")
        for i, p in enumerate(qa_pairs[:5]):
            print(f"\n[{i+1}] Q: {p['question'][:160]}")
            print(f"     A: {p['response'][:240]}")
        log.info("(Total matches: %d, not saved)", len(qa_pairs))
        return

    save_qa(qa_pairs)
    ingest_to_chroma(qa_pairs)

    log.info("")
    log.info("Done. Restart start-everything.bat to pick up the enriched Chroma index.")


if __name__ == "__main__":
    main()
