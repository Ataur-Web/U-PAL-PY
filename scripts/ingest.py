"""
corpus ingestion pipeline - converts raw json knowledge sources into chromadb vectors

this script loads multiple knowledge sources and chunks them into the vector database:
- uwtsd corpus: 336+ scraped pages with verified uwtsd.ac.uk source urls
- facts: curated uwtsd facts and policies
- knowledge: intent-driven responses for common student questions
- welsh bootstrap: academic terminology for welsh-language queries

all documents are split, embedded, indexed, and stored with full source metadata.
sources are ranked by bm25 (keyword) + dense embeddings (semantic similarity).

usage:
  python -m scripts.ingest         # add new docs to existing index
  python -m scripts.ingest --reset # drop and rebuild entire collection
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


# a split can leave fragments that carry no information: the scraper's text
# extraction produces chunks such as ";" or ", MSc" from list markup and table
# cells. these are actively harmful rather than merely useless - a near-empty
# string embeds to a vector that sits spuriously close to any short query, so
# they surfaced as the top dense hit for "thanks", "wow" and "ok" alike, and
# were then shown to students as cited sources with real UWTSD urls attached.
#
# corpus chunks average 375 characters, so requiring 40 letters and 6 words
# discards the fragments without touching genuine content. the filter is
# applied only to split corpus text; the Welsh bootstrap entries are short by
# design ("cwrs (Saesneg: course)") and are not passed through it.
_MIN_CHUNK_LETTERS = 40
_MIN_CHUNK_WORDS = 6


def _is_substantive(text: str) -> bool:
    letters = sum(1 for c in text if c.isalnum())
    words = [w for w in text.split() if any(c.isalnum() for c in w)]
    return letters >= _MIN_CHUNK_LETTERS and len(words) >= _MIN_CHUNK_WORDS


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
    dropped = 0
    for i, e in enumerate(entries):
        if not isinstance(e, dict):
            continue
        text = (e.get("text") or e.get("content") or "").strip()
        if not text:
            continue
        title = e.get("title") or e.get("heading") or f"corpus_{i}"
        # Extract actual URL from corpus (for verified source attribution)
        source_url = e.get("url") or e.get("source") or "uwtsd-corpus"
        for chunk in SPLITTER.split_text(text):
            if not _is_substantive(chunk):
                dropped += 1
                continue
            docs.append(Document(
                page_content=chunk,
                metadata={
                    "title":  title,
                    "source": source_url,
                    "lang":   _detect_lang(chunk),
                },
            ))
    log.info("corpus -> %d chunks (dropped %d empty fragments)", len(docs), dropped)
    return docs


def _load_facts() -> list[Document]:
    """Load the curated fact entries, one Document per language.

    uwtsd-facts.json stores each fact as
        {"id", "questions", "answer_en", "answer_cy", "keywords", "source_url"}
    An earlier version of this loader read "topic" and "fact"/"answer", keys
    which the file has never contained, so every entry failed the emptiness
    check and the whole layer was silently dropped from the index. The bug was
    invisible because the loader logged "facts -> 0 docs" and ingestion carried
    on with the remaining sources.

    The answer text is used as page_content rather than the question list, so
    the excerpt surfaced in the source panel reads as an answer. Facts are
    short by construction and are not passed through the splitter.
    """
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
        fid = e.get("id") or f"fact_{i}"
        # "campus-sa1-location" -> "Campus SA1 Location"
        title = " ".join(
            w.upper() if len(w) <= 3 and any(c.isdigit() for c in w) else w.capitalize()
            for w in str(fid).replace("-", " ").replace("_", " ").split()
        )
        # entries with no crawled equivalent keep the internal marker rather
        # than a guessed url. see scripts/map_source_urls.py
        source = e.get("source_url") or "uwtsd-facts"

        for lang_tag, key in (("en", "answer_en"), ("cy", "answer_cy")):
            text = (e.get(key) or "").strip()
            if not text:
                continue
            docs.append(Document(
                page_content=text,
                metadata={
                    "title":  title,
                    "source": source,
                    "lang":   lang_tag,
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
        # institutional page this intent's content derives from, assigned and
        # validated against the crawl by scripts/map_source_urls.py. purely
        # conversational intents (greeting, thanks) and intents covering
        # authenticated systems with no public page keep the internal marker.
        source = entry.get("source_url") or "knowledge"

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
                        "source": source,
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


_BYDTERMCYMRU_URL = "https://termau.cymru/"


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
                # these terms come from BydTermCymru, the Welsh Government
                # terminology service, not from UWTSD. pointing at the real
                # originating service makes them externally verifiable while
                # keeping them distinguishable from institutional sources in
                # the evaluation.
                "source": _BYDTERMCYMRU_URL,
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
