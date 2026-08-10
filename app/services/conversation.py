"""conversation state. we pull out the student's profile and the topic
they're talking about so the prompt can personalise the reply.

given the current message + recent history we return a dict like:
    {
      "name":        "Ataur",
      "course":      "computing",
      "level":       "undergraduate",
      "year":        "second",
      "campus":      "swansea",
      "topic_stack": ["accommodation", "fees"],
      "anchor":      "computing"      # most recently mentioned concrete topic
    }

we use regex patterns rather than a trained model because:
  - the signals are closed-set (UWTSD courses, the 4 campuses, etc)
  - a regex is transparent and easy to defend in the viva
  - no training data required

ref: Jurafsky, D. and Martin, J.H. (2024) Speech and Language Processing,
     3rd ed., chapter on information extraction (slot filling)
"""
from __future__ import annotations

import re
from typing import Any


# used to pull a first name from "I'm Ataur" / "my name is Ataur".
# we require a capitalised first letter to avoid matching verbs like "I'm going".
_NAME_RE = re.compile(r"\b(?:i'?m|my\s+name\s+is|i\s+am)\s+([A-Z][a-zA-Z'-]{1,20})\b")

# UWTSD course list, mapped to a canonical label. we match both Welsh and
# English where relevant.
_COURSE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(applied\s+computing|computer\s+science|computing)\b", re.I), "computing"),
    (re.compile(r"\b(cyber|cybersecurity)\b", re.I),                              "cyber security"),
    (re.compile(r"\b(games\s+design|game\s+dev|game\s+development)\b", re.I),     "games"),
    (re.compile(r"\b(nursing|midwifery)\b", re.I),                                "nursing"),
    (re.compile(r"\b(business|management|mba)\b", re.I),                          "business"),
    (re.compile(r"\b(psychology)\b", re.I),                                       "psychology"),
    (re.compile(r"\b(engineering|mechanical|electrical|civil|automotive)\b", re.I), "engineering"),
    (re.compile(r"\b(art|fine\s+art|graphic\s+design|fashion)\b", re.I),          "art & design"),
    (re.compile(r"\b(film|acting|performing\s+arts|drama)\b", re.I),              "performing arts"),
    (re.compile(r"\b(education|pgce|teacher|teaching)\b", re.I),                  "education"),
    (re.compile(r"\b(law|criminology)\b", re.I),                                  "law"),
    (re.compile(r"\b(architecture|construction)\b", re.I),                        "architecture"),
    (re.compile(r"\b(sport|fitness|coaching)\b", re.I),                           "sport"),
]

_LEVEL_RE = re.compile(
    r"\b(undergrad(uate)?|postgrad(uate)?|foundation|masters?|phd|mba|pgce)\b",
    re.I,
)
_YEAR_RE = re.compile(r"\b(first|second|third|fourth|final|1st|2nd|3rd|4th)\s+year\b", re.I)
_CAMPUS_RE = re.compile(
    r"\b(swansea|carmarthen|lampeter|cardiff|sa1|mount\s+pleasant|townhill)\b",
    re.I,
)

# topic the user is currently talking about. used to build topic_stack
# which the LLM prompt checks when resolving pronouns ("is it expensive?").
_TOPIC_SIGNALS: list[tuple[re.Pattern, str]] = [
    # "courses" had no signal at all, so "what courses are available?" left an
    # empty topic stack and a follow-up "is it expensive?" had nothing to
    # resolve against. \b stops these matching "coursework", which belongs to
    # assessment below.
    # the Welsh forms include their soft-mutated variants: initial c- becomes
    # g- after certain words, so "pa gyrsiau sydd ar gael?" carries "gyrsiau"
    # rather than "cyrsiau", and gradd- becomes radd-. matching only the radical
    # form silently misses the most natural phrasing of the question. this is
    # the incomplete mutation handling noted in the Welsh NLP literature.
    (re.compile(
        r"\b(course|courses|programme|programmes|degree|degrees|"
        r"cwrs|cyrsiau|gwrs|gyrsiau|gradd|graddau|radd|raddau)\b", re.I,
    ), "courses"),
    (re.compile(r"\b(accommodation|halls|residence|neuadd|llety)\b", re.I), "accommodation"),
    # plurals matter here: \bfee\b does not match "fees", which is how students
    # actually phrase it. same for costs and bursaries. ffi/ffioedd are the
    # Welsh singular and plural.
    (re.compile(
        r"\b(fee|fees|tuition|cost|costs|price|prices|"
        r"bursary|bursaries|scholarship|scholarships|ffi|ffioedd)\b", re.I,
    ), "fees"),
    (re.compile(r"\b(library|llyfrgell|study\s+space)\b", re.I),            "library"),
    (re.compile(r"\b(wellbeing|stress|mental|support|counsel)\b", re.I),    "wellbeing"),
    (re.compile(r"\b(wifi|moodle|mytsd|password|login|portal)\b", re.I),    "it"),
    (re.compile(r"\b(campus|carmarthen|lampeter|swansea|cardiff)\b", re.I), "campus"),
    (re.compile(r"\b(apply|application|ucas|offer|entry|clearing)\b", re.I), "admissions"),
    (re.compile(r"\b(graduation|ceremony|graddio)\b", re.I),                "graduation"),
    (re.compile(r"\b(placement|internship|work\s+experience)\b", re.I),     "placement"),
    (re.compile(r"\b(society|club|union|sport)\b", re.I),                   "student life"),
    (re.compile(r"\b(visa|international|tier\s+4)\b", re.I),                "international"),
    (re.compile(r"\b(timetable|lecture|class|module)\b", re.I),             "academics"),
    (re.compile(r"\b(exam|assessment|coursework|deadline)\b", re.I),        "assessment"),
    (re.compile(r"\b(job|career|employability)\b", re.I),                   "careers"),
]


def _first_match(rx: re.Pattern, text: str) -> str | None:
    m = rx.search(text)
    return m.group(0).lower() if m else None


def _detect_course(text: str) -> str | None:
    for rx, label in _COURSE_PATTERNS:
        if rx.search(text):
            return label
    return None


def _detect_topics(text: str) -> list[str]:
    # returns topics in first-seen order, no duplicates
    found: list[str] = []
    for rx, label in _TOPIC_SIGNALS:
        if rx.search(text) and label not in found:
            found.append(label)
    return found


def _find_name(history: list[dict[str, Any]]) -> str | None:
    # walk through history to find the first "I'm X" style introduction
    for turn in history:
        if turn.get("role") != "user":
            continue
        m = _NAME_RE.search(turn.get("text") or "")
        if m:
            return m.group(1)
    return None


# how many past user turns count as "recent" when resolving a reference.
# beyond this the topic is treated as closed.
_HISTORY_WINDOW = 2

# a message that refers back to something rather than naming it. only these
# inherit the previous turn's topic. english and welsh anaphora plus the
# common continuation words ("more", "else", "mwy").
_ANAPHORA_RE = re.compile(
    r"\b(it|its|it's|that|this|they|them|those|these|one|there|"
    r"more|another|else|both|"
    r"hwn|hwnnw|hynny|honno|nhw|fe|hi|mwy|arall|hefyd)\b",
    re.I,
)


def is_anaphoric(message: str) -> bool:
    return bool(_ANAPHORA_RE.search(message or ""))


def mentions_topic(message: str) -> bool:
    """True when the message names a subject the chatbot can answer about.

    Used to keep genuine questions out of the conversational path: a message
    that mentions courses, fees, accommodation and so on is asking about
    something, however briefly it is phrased.
    """
    return bool(_detect_topics(message or ""))


def build_state(message: str, history: list[dict[str, Any]]) -> dict[str, Any]:
    user_turns = [h for h in history if h.get("role") == "user"]

    # DURABLE profile signals. a student who says "I'm a second-year Computing
    # student" is still one ten turns later, so these are read from the whole
    # conversation.
    combined_user = " ".join((h.get("text") or "") for h in user_turns) + " " + message

    # TRANSIENT conversational focus. this previously came from the same
    # concatenation, which meant a topic mentioned once was never released:
    # after "what courses are available?", every later turn - "thanks", "wow",
    # even "where is the SA1 campus?" - still carried topic=courses, and the
    # prompt duly steered the model back to courses every time.
    #
    # topics now come from the current message. history is consulted only when
    # the message actually refers back to something ("is it expensive?"), and
    # then only within a short window.
    current_topics = _detect_topics(message)
    if current_topics:
        topic_stack = current_topics
    elif is_anaphoric(message):
        window = " ".join(
            (h.get("text") or "") for h in user_turns[-_HISTORY_WINDOW:]
        )
        topic_stack = _detect_topics(window)
    else:
        topic_stack = []

    # the anchor is the concrete thing "it" would refer to. same rule: prefer
    # what the current message names, fall back only for genuine references.
    anchor = _detect_course(message)
    if not anchor and current_topics:
        anchor = current_topics[0]
    if not anchor and is_anaphoric(message):
        for t in reversed(user_turns[-_HISTORY_WINDOW:]):
            text = t.get("text") or ""
            course = _detect_course(text)
            if course:
                anchor = course
                break
            topics = _detect_topics(text)
            if topics:
                anchor = topics[0]
                break

    return {
        "name":        _find_name(history),
        "course":      _detect_course(combined_user),
        "level":       _first_match(_LEVEL_RE, combined_user),
        "year":        _first_match(_YEAR_RE, combined_user),
        "campus":      _first_match(_CAMPUS_RE, combined_user),
        "topic_stack": topic_stack,
        "anchor":      anchor,
    }
