"""Welsh detection and bilingual query augmentation.

this is the language layer. ported from the Node version in u-pal-rag and
extended with BydTermCymru terminology.

the data files (in app/data/) are:
  welsh-terms.json          Welsh vocabulary (~43k words)
  welsh-bilingual-map.json  bilingual pairs, Welsh to English
  termcymru-terms.json      extra domain-specific pairs from BydTermCymru
                            (run scripts/fetch_termcymru.py to generate)

two public jobs:
  detect_language(text)  returns "en" or "cy"
  augment_query(text, lang)  adds English glosses to Welsh queries so the
                             English-indexed vector store still matches

ref: BydTermCymru - https://termau.cymru/
ref: Prys, D. et al. (2019) Welsh language technology, Language Resources
     and Evaluation Conference
"""
from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from pathlib import Path

from app.config import DATA_DIR


log = logging.getLogger("u-pal-py.welsh")

# orthographic tells. things like double-l, double-d, the circumflex
# vowels, and initial consonant clusters (ll/rh/ff/dd) appear in some
# English words ("hello", "entry") so we anchor most of them to a word
# boundary to avoid false positives. circumflex vowels are a strong
# language-unique signal so they don't need anchoring.
# ref: Thomas, P.W. (1996) Gramadeg y Gymraeg, University of Wales Press
_WELSH_ORTHO = re.compile(
    r"("
    # circumflex vowels are Welsh-only in normal web text
    r"[âêîôûŵŷ]"
    # initial double consonants only when they start a word
    r"|\b(?:ll|rh|dd|ff|ngh|mh|nh)"
    # standalone Welsh function words (stricter than just substring match)
    r"|\b(?:gwyr|gwybod|ydyn|mewn|wyf|wyd|yr|yng|ym)\b"
    r")",
    re.IGNORECASE,
)

# circumflex vowels are Welsh-only in typical web text
_WELSH_CHARS = re.compile(r"[âêîôûŵŷ]", re.IGNORECASE)

# word-tokeniser. we just match runs of letters + apostrophes, which is
# enough for UWTSD chat input, it doesn't need to be a full NLP tokeniser.
_TOKEN_RE = re.compile(r"[a-zâêîôûŵŷ']+", re.IGNORECASE)

# high-frequency Welsh words that don't exist as common English words.
# we exclude single letters (i, o, a, y) and English homographs (hi, na,
# no, on) because they cause English queries to be misclassified as
# Welsh. a single hit on the words below is enough to flip the decision
# because they are unambiguously Welsh.
_WELSH_FUNCTION_WORDS: frozenset[str] = frozenset({
    # articles and prepositions (Welsh-specific spellings only)
    "yr", "yn", "yng", "ym", "drwy", "wrth",
    # pronouns (English homographs like "hi", "fe", "fo" removed)
    "ti", "ni", "chi", "nhw", "eu", "ein", "dy", "fy", "eich",
    # common verbs and copulas
    "mae", "oes", "ydy", "yw", "oedd", "roedd", "bydd", "fydd",
    "gall", "gallaf", "hoffwn", "hoffaf", "eisiau", "moyn",
    "dw", "dwi", "rwy", "rydw", "rydych", "maen",
    # question words
    "beth", "pwy", "ble", "pryd", "pam", "sut", "faint", "sawl",
    # conjunctions and particles (English homographs removed)
    "ac", "ond", "achos", "oherwydd", "hefyd", "nawr",
    "dim", "nid", "nad",
    # uni-specific nouns that a UWTSD student might type
    "prifysgol", "coleg", "myfyriwr", "myfyrwyr", "darlith", "cwrs",
    "modiwl", "gradd", "campws", "llety", "ffioedd",
    # greetings (English homographs like "da" and "nos" removed)
    "shwmae", "shwmai", "prynhawn", "dda",
})


# common English words that overlap with the Welsh vocab. without this
# list a sentence like "What are the entry requirements?" gets a single
# vocab hit on "what" (which has a Welsh meaning too), trips the 20%
# ratio threshold, and is classified as Welsh. these tokens never count
# towards the Welsh hit total.
_ENGLISH_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "of", "in", "on", "at",
    "to", "for", "with", "from", "by", "as", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "should", "could", "may", "might", "must", "can",
    "i", "you", "he", "she", "it", "we", "they", "me", "him", "her",
    "us", "them", "my", "your", "his", "its", "our", "their",
    "this", "that", "these", "those",
    "what", "when", "where", "why", "who", "how", "which",
    "no", "not", "yes", "all", "any", "some", "more", "most", "less",
    "hello", "hi", "hey", "thanks", "please", "ok", "okay",
    "course", "courses", "fee", "fees", "module", "modules",
    "campus", "library", "student", "students", "uni", "university",
    "help", "tell", "show", "give", "make", "take", "get", "go", "come",
    "find", "see", "know", "think", "want", "need", "like", "feel",
    "today", "tomorrow", "yesterday", "now", "then", "here", "there",
    "good", "bad", "new", "old", "first", "last", "next", "previous",
    "year", "years", "day", "days", "week", "weeks", "month", "months",
    "time", "money", "people", "person", "thing", "things", "way", "ways",
    "about", "after", "before", "between", "during", "into", "over",
    "through", "under", "until", "while", "without",
    "deadline", "exam", "exams", "essay", "essays", "assignment",
    "assignments", "lecture", "lectures", "tutor", "tutors",
    "computing", "engineering", "business", "law", "art", "design",
})


# we keep loaded data on a class-like struct so the module has one
# clear place for state instead of a bag of global dicts.
class _State:
    vocab: set[str] = set()            # all Welsh words we know about
    bilingual: dict[str, str] = {}     # cy -> en
    reverse:   dict[str, str] = {}     # en -> cy
    termcymru_count: int = 0           # how many pairs came from BydTermCymru
    loaded: bool = False


_state = _State()


def _load_json(path: Path) -> object | None:
    # tiny helper so we don't repeat the try/except pattern for every file
    if not path.exists():
        log.warning("Welsh data file missing: %s", path)
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        log.error("Failed to load %s: %s", path, e)
        return None


def load() -> None:
    # idempotent: safe to call multiple times, it only reads files once
    if _state.loaded:
        return

    # welsh-terms.json. either a plain list or {"terms":[...]}
    terms = _load_json(DATA_DIR / "welsh-terms.json")
    if isinstance(terms, dict):
        terms = terms.get("terms") or terms.get("words") or []
    if isinstance(terms, list):
        _state.vocab = {t.strip().lower() for t in terms if isinstance(t, str) and t.strip()}

    # inner helper that handles both list-of-pairs and plain dict shapes
    # for bilingual maps, so we can reuse it across multiple files.
    def _ingest_bmap(bmap: object) -> None:
        if isinstance(bmap, list):
            for pair in bmap:
                if not isinstance(pair, dict):
                    continue
                cy = (pair.get("cy") or pair.get("welsh") or "").strip().lower()
                en = (pair.get("en") or pair.get("english") or "").strip().lower()
                if cy and en:
                    # setdefault so earlier entries win on duplicates
                    _state.bilingual.setdefault(cy, en)
                    _state.reverse.setdefault(en, cy)
                    _state.vocab.add(cy)
        elif isinstance(bmap, dict):
            for cy, en in bmap.items():
                if isinstance(cy, str) and isinstance(en, str):
                    _state.bilingual.setdefault(cy.lower(), en.lower())
                    _state.reverse.setdefault(en.lower(), cy.lower())
                    _state.vocab.add(cy.lower())

    bmap = _load_json(DATA_DIR / "welsh-bilingual-map.json")
    if bmap is not None:
        _ingest_bmap(bmap)

    # extra BydTermCymru pairs, optional (you have to run the fetch script)
    tc_path = DATA_DIR / "termcymru-terms.json"
    if tc_path.exists():
        tc = _load_json(tc_path)
        before = len(_state.bilingual)
        if tc is not None:
            _ingest_bmap(tc)
        _state.termcymru_count = len(_state.bilingual) - before
        log.info("[Welsh] +%d pairs from BydTermCymru", _state.termcymru_count)

    _state.loaded = True
    log.info(
        "[Welsh] Loaded %d vocab terms, %d bilingual pairs (%d from BydTermCymru)",
        len(_state.vocab), len(_state.bilingual), _state.termcymru_count,
    )


# load at import time so the first request doesn't pay the cost
load()


def is_loaded() -> bool:
    return _state.loaded


def vocab_size() -> int:
    return len(_state.vocab)


def bilingual_map_size() -> int:
    return len(_state.bilingual)


# we cache because detect_language is called on every request and the
# regex scans aren't free. 4096 is enough for a dev session's worth of
# unique inputs.
@lru_cache(maxsize=4096)
def looks_welsh(text: str) -> bool:
    if not text:
        return False
    if _WELSH_CHARS.search(text):
        return True
    return bool(_WELSH_ORTHO.search(text))


def detect_language(text: str) -> str:
    """return "cy" or "en".

    we combine multiple signals rather than relying on a single model:

      1. Welsh-only Unicode chars (â, ê, î, ô, û, ŵ, ŷ)
      2. Welsh orthographic patterns (ll, dd, rh, ff, ngh, ...)
      3. high-frequency Welsh function words (mae, beth, prifysgol, ...)
         a single strong function word is enough
      4. vocab hit-rate >= 20% against the full vocab (for 4+ token text)
      5. any single vocab hit for very short queries

    using several weak signals together is more reliable for the short
    messages students type than a single threshold. tested against
    the Node baseline on the same test inputs, see tests/test_welsh.py.
    ref: adapted from u-pal-rag/lib/nlp.js and extended
    """
    if not text or not text.strip():
        return "en"

    if _WELSH_CHARS.search(text):
        return "cy"

    if _WELSH_ORTHO.search(text):
        return "cy"

    tokens = [t.lower() for t in _TOKEN_RE.findall(text)]
    if not tokens:
        return "en"

    # signal 3, any single Welsh function word decides it
    if any(t in _WELSH_FUNCTION_WORDS for t in tokens):
        return "cy"

    if not _state.vocab:
        return "en"

    # only count vocab hits that are NOT also common English words. this
    # stops English questions like "What are the entry requirements?"
    # from being misclassified as Welsh because "what" happens to have
    # a Welsh entry in the bilingual map.
    hits = sum(
        1 for t in tokens
        if t in _state.vocab and t not in _ENGLISH_STOPWORDS
    )

    # signal 4/5, vocab coverage. for very short queries we require at
    # least 2 hits (or one diacritic / orthography signal already would
    # have returned cy above). single-hit cy detection caused English
    # words like "hi" or "no" to flip the language because the bilingual
    # map contains them as Welsh pronouns / particles.
    if len(tokens) < 4:
        return "cy" if hits >= 2 else "en"

    ratio = hits / len(tokens)
    # 0.20 was picked by eyeballing the test set. the Node version
    # used 0.25 but that missed some valid Welsh queries.
    return "cy" if ratio >= 0.20 else "en"


def augment_query(text: str, lang: str) -> str:
    # for Welsh queries we append English glosses of known terms. the
    # vector store is indexed mostly in English, so this helps recall.
    # we tag the extras with [en: ...] so they're easy to strip back out.
    if lang != "cy" or not _state.bilingual or not text:
        return text

    seen: set[str] = set()
    extras: list[str] = []
    for tok in _TOKEN_RE.findall(text.lower()):
        if tok in seen:
            continue
        seen.add(tok)
        gloss = _state.bilingual.get(tok)
        if gloss:
            extras.append(gloss)

    if not extras:
        return text
    return f"{text}  [en: {' '.join(extras)}]"


def translate_welsh_to_english(text: str) -> str:
    # best-effort word-by-word gloss. only used by the retriever, never
    # shown to the user (the real translation goes through the LLM).
    if not _state.bilingual:
        return text

    def _swap(match: re.Match) -> str:
        tok = match.group(0)
        return _state.bilingual.get(tok.lower(), tok)

    # re.sub with a function lets us keep the original punctuation/spacing
    # ref: https://docs.python.org/3/library/re.html#re.sub
    return _TOKEN_RE.sub(_swap, text)
