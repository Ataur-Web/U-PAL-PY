"""knowledge-base intent classifier.

this is a lightweight port of the NLP intent matcher from u-pal-rag/lib/nlp.js.
every entry in knowledge.json has a `tag` and a list of `patterns`. we
fuzzy-match the user's message against every pattern, take the max score
per intent, then apply a small history boost so follow-up questions in the
same topic get ranked higher.

ref: Jurafsky, D. and Martin, J.H. (2024) Speech and Language Processing,
     3rd ed., chapter on dialogue systems (intent classification)
ref: rapidfuzz library for fuzzy string matching - https://github.com/maxbachmann/RapidFuzz
"""
from __future__ import annotations

import json
import logging
import re
from functools import lru_cache
from typing import Any

from rapidfuzz import fuzz

from app.config import DATA_DIR


log = logging.getLogger("u-pal-py.intent")


@lru_cache(maxsize=1)
def _load_knowledge() -> list[dict[str, Any]]:
    # cached because the file is a few hundred KB and we don't want to
    # re-parse it on every request.
    path = DATA_DIR / "knowledge.json"
    if not path.exists():
        log.warning("knowledge.json missing at %s", path)
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        log.error("knowledge.json parse failed: %s", e)
        return []

    # handle both shapes. older files were wrapped as {"intents": [...]},
    # newer ones are just a list.
    entries = data.get("intents") if isinstance(data, dict) else data
    if not isinstance(entries, list):
        return []

    cleaned: list[dict[str, Any]] = []
    for e in entries:
        if not isinstance(e, dict):
            continue
        tag = e.get("tag") or e.get("name")
        patterns = e.get("patterns") or e.get("triggers") or []
        responses = e.get("responses") or []
        if tag and patterns:
            cleaned.append({"tag": tag, "patterns": patterns, "responses": responses})
    log.info("[Intent] Loaded %d knowledge entries", len(cleaned))
    return cleaned


# boost tables. these mirror the Node version.
# SPECIFIC_TERMS picks out particular UWTSD courses the user mentioned
# earlier ("I'm studying Computing"), so when they ask a follow-up we
# bias towards that course's tag.
SPECIFIC_TERMS: list[tuple[re.Pattern, list[str]]] = [
    (re.compile(r"\b(applied\s+computing|computer\s+science|computing|cyber|games)\b", re.I),
        ["comput", "cyber", "games"]),
    (re.compile(r"\b(nursing|midwifery|healthcare)\b", re.I), ["nursing", "health"]),
    (re.compile(r"\b(business|management|mba|marketing|finance|accounting)\b", re.I),
        ["business", "mba"]),
    (re.compile(r"\b(psychology|counselling|social\s+work)\b", re.I),
        ["psychology", "counsell", "social_work"]),
    (re.compile(r"\b(engineering|mechanical|electrical|civil|automotive)\b", re.I), ["engineer"]),
    (re.compile(r"\b(art|design|graphic|fine\s+art|fashion)\b", re.I), ["art", "design"]),
    (re.compile(r"\b(film|acting|performing\s+arts|drama|theatre)\b", re.I),
        ["film", "acting", "performing"]),
    (re.compile(r"\b(education|teacher|pgce|teaching)\b", re.I), ["education", "pgce", "teach"]),
    (re.compile(r"\b(law|criminology)\b", re.I), ["law", "crimin"]),
    (re.compile(r"\b(architecture|construction|built\s+environment)\b", re.I),
        ["architect", "construct"]),
    (re.compile(r"\b(sport|fitness|coaching)\b", re.I), ["sport", "fitness"]),
]

# FAMILY_HINTS is broader, a question mentioning "accommodation" boosts all
# accommodation-related intents regardless of course.
FAMILY_HINTS: dict[str, re.Pattern] = {
    "accommodation": re.compile(r"\b(accommodation|halls|residence|neuadd|llety)\b", re.I),
    "fees":          re.compile(r"\b(fee|tuition|cost|price|ffioedd|scholarship|bursary)\b", re.I),
    "wellbeing":     re.compile(r"\b(wellbeing|stress|anxious|mental|lles|support|crisis)\b", re.I),
    "it":            re.compile(r"\b(wifi|moodle|mytsd|password|login|portal)\b", re.I),
    "library":       re.compile(r"\b(library|llyfrgell|book|study\s+space|founders)\b", re.I),
    "campus":        re.compile(r"\b(campus|carmarthen|lampeter|swansea|sa1|townhill|cardiff)\b", re.I),
    "apply":         re.compile(r"\b(apply|application|ucas|offer|entry\s+requirement|clearing)\b", re.I),
    "graduation":    re.compile(r"\b(graduation|graduating|ceremony|graddio)\b", re.I),
}


def _score_entry(message: str, entry: dict[str, Any]) -> float:
    # token_set_ratio handles word-order differences and gives a score
    # 0-100. we take the max over all patterns for this intent.
    # ref: https://github.com/rapidfuzz/RapidFuzz#token-set-ratio
    best = 0.0
    for p in entry.get("patterns", []):
        s = fuzz.token_set_ratio(message, p)
        if s > best:
            best = float(s)
    return best


def _apply_history_boost(
    candidates: list[tuple[dict[str, Any], float]],
    history: list[dict[str, Any]],
) -> list[tuple[dict[str, Any], float]]:
    # we look at the last 4 turns only. older context is usually stale
    # and adds noise to the boost.
    if not history:
        return candidates

    recent_text = " ".join(
        (h.get("text") or "").lower()
        for h in history[-4:]
        if isinstance(h, dict)
    )
    if not recent_text:
        return candidates

    # which specific courses / families were mentioned recently?
    active_specific: list[str] = []
    for rx, tags in SPECIFIC_TERMS:
        if rx.search(recent_text):
            active_specific.extend(tags)

    active_families = [
        fam for fam, rx in FAMILY_HINTS.items() if rx.search(recent_text)
    ]

    boosted: list[tuple[dict[str, Any], float]] = []
    for entry, score in candidates:
        tag = entry["tag"].lower()
        # tier 1: same course match, big boost
        if active_specific and any(t in tag for t in active_specific):
            boosted.append((entry, score * 2.5))
            continue
        # tier 1 penalty: different course family, dampen so we don't
        # confuse "tell me more about Computing" with Nursing intents
        if active_specific and tag.startswith("courses_") \
                and not any(t in tag for t in active_specific):
            boosted.append((entry, score * 0.3))
            continue
        # tier 2: family hint, moderate boost
        if any(fam in tag for fam in active_families):
            boosted.append((entry, score * 1.5))
            continue
        boosted.append((entry, score))
    return boosted


def classify(
    message: str,
    lang: str = "en",                 # noqa: ARG001  (reserved for future CY tuning)
    history: list[dict[str, Any]] | None = None,
    min_score: float = 45.0,
) -> dict[str, Any] | None:
    # returns the top-scoring knowledge entry or None if nothing crosses
    # the threshold. min_score 45 came from eyeballing scores on the
    # test set, see tests/test_intent_emotion.py.
    entries = _load_knowledge()
    if not entries or not message:
        return None

    scored = [(e, _score_entry(message, e)) for e in entries]
    scored = _apply_history_boost(scored, history or [])
    scored.sort(key=lambda t: t[1], reverse=True)

    if not scored or scored[0][1] < min_score:
        return None

    best_entry, best_score = scored[0]
    return {**best_entry, "score": best_score}
