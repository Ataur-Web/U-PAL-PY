"""one-off script that builds the Chroma vector index from the JSON
corpora living in app/data/. also bootstraps a slice of academic Welsh
terminology so the cy-filtered retrieval pass has content from day one.

Usage:
    python -m scripts.ingest              # incremental add
    python -m scripts.ingest --reset      # drop and rebuild

Inputs (any that exist are loaded):
    app/data/uwtsd-corpus.json     list of { title, text } passages
    app/data/uwtsd-facts.json      list of { topic, fact } entries
    app/data/knowledge.json        intents to stringified responses
    app/data/welsh-bilingual-map.json  bilingual map for the cy bootstrap
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import DATA_DIR
from app.services import rag, welsh


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("ingest")


# sentence-aware splitter tuned for the UWTSD corpus. we use a smaller
# chunk_size than the LangChain default because the embedding model is
# MiniLM-L12 (384 dims, ~512 token context), and short focused chunks
# get higher cosine similarity on short student questions than long
# rambling ones. the separator list covers both English and Welsh
# sentence boundaries (Welsh uses the same ., ?, ! punctuation).
# ref: https://python.langchain.com/docs/how_to/recursive_text_splitter/
# ref: Sentence-BERT (Reimers & Gurevych, 2019) on chunk size sensitivity
SPLITTER = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=80,
    length_function=len,
    is_separator_regex=False,
    separators=[
        "\n\n",   # paragraph break, strongest semantic boundary
        "\n",     # line break
        ". ",     # English / Welsh sentence end
        "? ",     # question (both languages)
        "! ",     # exclamation (both languages)
        "; ",     # clause break
        ", ",     # weakest fallback that still respects punctuation
        " ",      # last resort, word boundary
        "",       # absolute last resort
    ],
)


def _detect_lang(text: str) -> str:
    # tag each chunk with its language so the cy-filter in rag.retrieve
    # can find Welsh passages when a Welsh query lands. uses the same
    # multi-signal detector the chat route runs.
    try:
        return welsh.detect_language(text)
    except Exception:
        return "en"


# loaders, each returns a list[Document] ready for the splitter

def _load_corpus() -> list[Document]:
    path = DATA_DIR / "uwtsd-corpus.json"
    if not path.exists():
        log.warning("skip: %s not found", path)
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data if isinstance(data, list) else data.get("passages", [])
    docs: list[Document] = []
    for i, e in enumerate(entries):
        if not isinstance(e, dict):
            continue
        text = (e.get("text") or e.get("content") or "").strip()
        if not text:
            continue
        title = e.get("title") or e.get("heading") or f"corpus_{i}"
        for chunk in SPLITTER.split_text(text):
            docs.append(Document(
                page_content=chunk,
                metadata={
                    "title":  title,
                    "source": "uwtsd-corpus",
                    "lang":   _detect_lang(chunk),
                },
            ))
    log.info("corpus -> %d chunks", len(docs))
    return docs


def _load_facts() -> list[Document]:
    path = DATA_DIR / "uwtsd-facts.json"
    if not path.exists():
        log.warning("skip: %s not found", path)
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data if isinstance(data, list) else data.get("facts", [])
    docs: list[Document] = []
    for i, e in enumerate(entries):
        if not isinstance(e, dict):
            continue
        topic = e.get("topic") or f"fact_{i}"
        fact  = (e.get("fact") or e.get("answer") or "").strip()
        if not fact:
            continue
        docs.append(Document(
            page_content=fact,
            metadata={
                "title":  topic,
                "source": "uwtsd-facts",
                "lang":   _detect_lang(fact),
            },
        ))
    log.info("facts -> %d docs", len(docs))
    return docs


def _load_knowledge() -> list[Document]:
    """Flatten knowledge.json into one Document per intent response."""
    path = DATA_DIR / "knowledge.json"
    if not path.exists():
        log.warning("skip: %s not found", path)
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.get("intents") if isinstance(data, dict) else data
    docs: list[Document] = []
    for entry in (entries or []):
        if not isinstance(entry, dict):
            continue
        tag = entry.get("tag") or entry.get("name") or "intent"

        # the knowledge file may store responses two ways:
        #   1. a flat list of strings (legacy single-language)
        #   2. a {"en": [...], "cy": [...]} dict (current bilingual)
        # we handle both so we can tag the cy strings with lang=cy and
        # the en strings with lang=en. this is what powers the cy-first
        # retrieval pass for Welsh queries.
        responses = entry.get("responses") or []
        if isinstance(responses, dict):
            buckets = [
                ("en", responses.get("en") or []),
                ("cy", responses.get("cy") or []),
            ]
        else:
            # legacy list, detect language per string
            buckets = [(None, responses)]

        for tag_lang, items in buckets:
            for resp in items:
                text = resp if isinstance(resp, str) else json.dumps(resp, ensure_ascii=False)
                text = text.strip()
                if not text:
                    continue
                docs.append(Document(
                    page_content=text,
                    metadata={
                        "title":  tag,
                        "source": "knowledge",
                        "lang":   tag_lang or _detect_lang(text),
                    },
                ))
    log.info("knowledge -> %d docs", len(docs))
    return docs


# filter the welsh-bootstrap to academic, educational and general
# student-life vocabulary. original bootstrap dumped every entry in
# the bilingual map (3k+) which polluted semantic search ("Hello"
# matching "helo"). this expanded keyword set keeps roughly half the
# map, focused on anything a student might ask a tutor about.
_WELSH_BOOTSTRAP_KEYWORDS = re.compile(
    r"\b("
    # academic and university
    r"prifysgol|coleg|ysgol|myfyriwr|myfyrwyr|darlith|cwrs|cyrsiau|"
    r"modiwl|modiwlau|gradd|graddau|israddedig|ôl-raddedig|"
    r"thesis|traethawd|aseiniad|aseiniadau|astudio|astudiaeth|"
    r"ymchwil|adolygu|arholiad|arholiadau|asesiad|asesiadau|"
    r"darlithydd|tiwtor|athro|ysgolhaig|academaidd|hyfforddi|"
    r"diploma|tystysgrif|cymhwyster|cymwysterau|safon uwch|TGAU|"
    r"semestr|tymor|blwyddyn|gwers|gwersi|dosbarth|amserlen|"
    # subjects
    r"mathemateg|gwyddoniaeth|ffiseg|cemeg|bioleg|technoleg|"
    r"hanes|daearyddiaeth|economeg|gwleidyddiaeth|athroniaeth|"
    r"seicoleg|cymdeithaseg|saesneg|cymraeg|llenyddiaeth|iaith|"
    r"cyfrifiadureg|peirianneg|pensaernïaeth|busnes|rheolaeth|"
    r"meddyginiaeth|meddygon|nyrsio|cyfraith|cyfreithiol|"
    r"celf|cerdd|drama|ffilm|dylunio|chwaraeon|addysg|"
    # student life, admin, finance
    r"ffi|ffioedd|tâl|cost|ariannol|cyllid|"
    r"ysgoloriaeth|grant|benthyciad|bwrsariaeth|noddi|"
    r"llety|neuadd|campws|llyfrgell|adran|swyddfa|"
    r"lles|cymorth|cefnogaeth|cwnsela|iechyd|"
    r"ymgeisio|cais|UCAS|cynnig|mynediad|cofrestru|cofrestriad|"
    r"yrfa|gyrfa|lleoliad|interniaeth|graddedig|cyflogadwyedd|"
    r"cyflog|swydd|gwaith|profiad|gweithle|"
    # study skills and learning verbs
    r"dysgu|deall|esbonio|disgrifio|cymharu|trafod|dadl|"
    r"sgiliau|methodoleg|damcaniaeth|cysyniad|theori|"
    r"darllen|ysgrifennu|cyfeiriad|cyfeirio|llyfryddiaeth|"
    r"cynllunio|trefnu|paratoi|amser|amserlennu|"
    r"datblygu|gwella|hyfforddiant|ymarfer|"
    # communication and general info
    r"e-bost|ebost|ffôn|cyswllt|cysylltu|cysylltiad|gwybodaeth|"
    r"gwefan|porth|cofrestrydd|gweinyddol|gweinyddiaeth|"
    r"cyhoeddi|cyhoeddiad|hysbysu|hysbysiad|datganiad|"
    # technology and computing
    r"cyfrifiadur|gliniadur|meddalwedd|caledwedd|rhyngrwyd|"
    r"cyfrineiriau|cyfrinair|porwr|app|ffeiliau|data|"
    # places, geography, campuses
    r"adeilad|ystafell|labordy|caffi|bwyty|cantîn|"
    r"abertawe|caerfyrddin|llambed|caerdydd|cymru|cymro|cymraes|"
    r"trafnidiaeth|bws|trên|teithio|llwybr|"
    # everyday concepts students reference
    r"amser|diwrnod|wythnos|mis|dyddiad|"
    r"dechrau|gorffen|parhau|gohirio|trefniant|"
    r"problem|atebion|ateb|cwestiwn|cwestiynau|"
    r"pwysig|defnyddiol|angenrheidiol|opsiynol|"
    r"newydd|hen|dilyn|dewis|dewisiadau|"
    r"cyfle|cyfleoedd|her|heriau|llwyddiant|"
    # health and wellbeing
    r"meddyg|nyrs|ysbyty|salwch|iechyd|diogelwch|stres|gofid|"
    r"emosiynol|meddyliol|corfforol|cwsg|maeth|"
    # international and visa
    r"rhyngwladol|fisa|teithio|tramor|cartref|fy mamwlad|"
    # forms, documents, letters
    r"ffurflen|dogfen|llythyr|tystysgrif|adroddiad|llawlyfr|canllaw|"
    # courses by category
    r"hyfforddiant|prentisiaeth|astudiaethau|astudiaeth|israddedigol"
    r")\b",
    re.IGNORECASE,
)


def _load_welsh_bootstrap() -> list[Document]:
    """Bootstrap academic Welsh-tagged docs from the bilingual map.

    We filter the full bilingual map (3k+ entries) down to the slice
    relevant to a student chatbot: courses, subjects, fees, application,
    accommodation, study skills. Random everyday words like "helo" or
    "tywydd" are dropped because they pollute semantic search and
    cause English greetings to retrieve Welsh-tagged passages.
    """
    path = DATA_DIR / "welsh-bilingual-map.json"
    if not path.exists():
        log.warning("skip: %s not found", path)
        return []

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        log.error("welsh-bootstrap: bad JSON in %s: %s", path, e)
        return []

    pairs: list[tuple[str, str]] = []
    if isinstance(raw, dict):
        for cy, en in raw.items():
            if isinstance(cy, str) and isinstance(en, str):
                pairs.append((cy.strip(), en.strip()))
    elif isinstance(raw, list):
        for entry in raw:
            if not isinstance(entry, dict):
                continue
            cy = (entry.get("cy") or entry.get("welsh") or "").strip()
            en = (entry.get("en") or entry.get("english") or "").strip()
            if cy and en:
                pairs.append((cy, en))

    # noisy single-word entries to always exclude even if they accidentally
    # match the keyword regex. these are exactly the words that caused
    # the "Hello" -> "helo" false-match bug.
    _ALWAYS_SKIP = {
        "helo", "shwmae", "shwmai", "diolch", "croeso", "hwyl", "iawn",
        "bore", "prynhawn", "nos", "tywydd", "hyfryd", "drwg",
        "yfory", "heddiw", "ddoe", "neithiwr",
    }

    docs: list[Document] = []
    skipped = 0
    for cy, en in pairs:
        cy_lower = cy.lower().strip()
        if cy_lower in _ALWAYS_SKIP:
            skipped += 1
            continue

        # multi-word Welsh entries (phrases like "absenoldeb awdurdodedig"
        # or "asesiad cymheiriaid") are specific terminology and almost
        # always domain-relevant. we keep them all without re-checking
        # the keyword filter.
        is_phrase = len(cy.split()) > 1

        if not is_phrase:
            # single-word entries get the strict keyword filter to drop
            # noise like greetings, weather, generic everyday vocab.
            combined = f"{cy} {en}".lower()
            if not _WELSH_BOOTSTRAP_KEYWORDS.search(combined):
                skipped += 1
                continue

        # mini bilingual passage. the cy-tag is what lets the filtered
        # retrieval pass return these on Welsh queries.
        text = f"{cy} (Saesneg: {en})"
        docs.append(Document(
            page_content=text,
            metadata={
                "title":  cy[:80],
                "source": "welsh-bootstrap",
                "lang":   "cy",
            },
        ))

    log.info("welsh-bootstrap -> %d cy-tagged docs (skipped %d off-topic)",
             len(docs), skipped)
    return docs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="drop the collection first")
    args = parser.parse_args()

    if args.reset:
        rag.reset_collection()

    all_docs: list[Document] = []
    all_docs += _load_corpus()
    all_docs += _load_facts()
    all_docs += _load_knowledge()
    all_docs += _load_welsh_bootstrap()

    if not all_docs:
        log.error("Nothing to ingest, did you copy the JSON files into app/data/?")
        return 1

    added = rag.ingest_documents(all_docs)
    log.info("Done.  Added %d documents.", added)
    return 0


if __name__ == "__main__":
    sys.exit(main())
