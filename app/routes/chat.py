"""
post /api/chat - main bilingual chatbot endpoint for u-pal

this endpoint orchestrates the complete rag pipeline:
1. language detection (en/cy) on incoming message
2. injection filtering (blocks prompt attacks pre-llm)
3. query augmentation (welsh→english term mapping)
4. intent classification (student services, courses, etc)
5. emotion detection (adjusts response tone if stressed)
6. chromadb retrieval (hybrid dense+bm25 search)
7. llm generation (claude or ollama)
8. source attribution (extracts verified uwtsd links)

response includes verified sources users can click to confirm information.
"""
from __future__ import annotations

import logging
import re
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services import conversation, emotion, intent, llm, rag, welsh


router = APIRouter()
log = logging.getLogger("u-pal-py.chat")


# pre-LLM injection filter. these patterns catch the most common attacks
# before they reach the model so we save api tokens and guarantee a safe
# refusal. anything not on this list still hits the LLM normally where
# the system prompt (system_en.txt SCOPE AND REFUSALS section) handles it.
# defense in depth, not a sole protection layer.
_INJECTION_PATTERNS = [
    # prompt extraction
    re.compile(r"\b(system\s+prompt|your\s+(full|complete|entire)\s+(prompt|instructions))\b", re.I),
    re.compile(r"\b(reveal|share|show|tell\s+me|print|output|display)\s+(your|the)\s+(prompt|instructions|system)\b", re.I),
    re.compile(r"\b(ignore|disregard|forget)\s+(all|previous|prior)\s+(instructions|rules|prompts)\b", re.I),
    re.compile(r"\b(you\s+are\s+now|act\s+as|pretend\s+to\s+be)\s+(dan|do\s+anything\s+now)\b", re.I),
    re.compile(r"\b(jailbreak|developer\s+mode|admin\s+mode|god\s+mode)\b", re.I),
    re.compile(r"\b(grandmother|grandma|mum|mother|nan)\s+(used\s+to\s+)?(read|tell|recite|sing)\s+me\b", re.I),
    re.compile(r"previous\s+instruction\s+(has\s+been|is)\s+(cancelled|canceled|revoked|overridden)", re.I),

    # role confusion / code execution attempts
    re.compile(r"\b(pretend|act|behave|simulate|roleplay)\s+(as|to\s+be|that\s+you\s+are)\s+(a\s+|an\s+)?(python|javascript|node|bash|shell|interpreter|compiler)\b", re.I),
    re.compile(r"\bexecute\s+(this|the\s+following|this\s+code|the\s+code)\b", re.I),
    re.compile(r"\bopen\s*\(\s*['\"][^'\"]+\.(txt|py|js|json|env|conf)['\"]", re.I),

    # off-topic technical requests we explicitly refuse
    re.compile(r"\b(write|generate|give\s+me|create|build|make)\s+(a\s+|an\s+|me\s+)?(python|javascript|bash|shell|node)\s+(script|code|program|tool|function)\b.*\b(scrap|crawl|harvest|extract|download)\b", re.I),
    re.compile(r"\b(scrap(e|ing|er)?|crawl(ing|er)?)\s+(twitter|x\.com|linkedin|instagram|facebook|reddit|tiktok|youtube|google)\b", re.I),
    re.compile(r"\b(brute\s+force|password\s+crack|sql\s+injection|xss\s+payload|csrf\s+exploit|0day|zero\s+day)\b", re.I),
]

# Welsh-language injection patterns. fewer than the English set because
# the attack tooling online is overwhelmingly English, but we cover the
# obvious ones. Welsh consonant mutations are a nuisance, words can
# start with c/g/ch, p/b/ph, t/d/th depending on context. we match the
# unchanging root suffix (-ieithydd, -honglydd) instead of the full
# word so all mutated forms (cyfieithydd, gyfieithydd) match the same
# rule.
_INJECTION_PATTERNS_CY = [
    re.compile(r"anwybyddwch.*(cyfarwyddiadau|rheolau|prompt)", re.I),
    re.compile(r"(dangoswch|rhannwch|datgelwch).*(prompt|cyfarwyddiadau|system)", re.I),
    re.compile(r"esgus.*(ieithydd|honglydd|python|shell|bash)", re.I),
    re.compile(r"\b(sgrap|crawl)\w*\s+(twitter|x\.com|facebook|instagram)", re.I),
]


def _is_injection_attempt(message: str, lang: str) -> str | None:
    """Return the matching pattern label if this message looks adversarial.

    None means pass through to the LLM as normal. A non-None return
    triggers the deterministic refusal path that skips the LLM call.
    """
    if not message:
        return None
    for rx in _INJECTION_PATTERNS:
        if rx.search(message):
            return rx.pattern[:60]
    if lang == "cy":
        for rx in _INJECTION_PATTERNS_CY:
            if rx.search(message):
                return rx.pattern[:60]
    return None


# hardcoded refusal templates returned when the pre-LLM filter fires.
# kept short and same-language. no API call so they cannot drift.
_INJECTION_REFUSAL_EN = (
    "That's outside what I can help with as a UWTSD academic chatbot. "
    "I focus on student support, courses, applications, fees, accommodation, "
    "campus life, wellbeing, and general academic study help. "
    "If you have a question along those lines I'd be happy to help."
)
_INJECTION_REFUSAL_CY = (
    "Mae hynny y tu hwnt i'r hyn y gallaf helpu gydag fel chatbot academaidd "
    "PCYDDS. Rwy'n canolbwyntio ar gymorth i fyfyrwyr, cyrsiau, ceisiadau, "
    "ffioedd, llety, bywyd campws, lles, a chymorth astudio academaidd cyffredinol. "
    "Os oes gennych gwestiwn ar y llinellau hynny, byddwn yn falch o helpu."
)


# pydantic models let FastAPI validate the request body for us and
# generate OpenAPI docs automatically at /docs.
# ref: https://fastapi.tiangolo.com/tutorial/body/
class HistoryTurn(BaseModel):
    role: Literal["user", "assistant"]
    text: str


class ChatRequest(BaseModel):
    # max_length=4000 protects the LLM from very long prompts that would
    # blow the context window
    message:     str                       = Field(..., min_length=1, max_length=4000)
    runningLang: Literal["en", "cy"] | None = None
    history:     list[HistoryTurn]         = Field(default_factory=list)


class SourceRef(BaseModel):
    title: str           # Human-readable title, e.g., "Entry Requirements"
    category: str        # Category, e.g., "Admissions"
    excerpt: str         # First ~150 chars of content
    language: str        # "en" or "cy"
    url: str | None = None  # Optional direct link to source


class ChatResponse(BaseModel):
    reply:   str
    lang:    Literal["en", "cy"]
    intent:  str | None  = None
    emotion: str         = "neutral"
    sources: list[SourceRef] = Field(default_factory=list)


# intents that are pure conversation rather than information requests. these
# are answered directly from knowledge.json and never reach retrieval or the
# LLM, so they cannot inherit the previous turn's topic.
_CHITCHAT_TAGS = frozenset({
    "greeting", "thanks", "goodbye", "how_are_you", "capabilities",
})

# deliberately stricter than the default 45. we only bypass the normal
# pipeline when the match is unambiguous; a borderline score means the message
# probably carries real content and should be answered properly.
_CHITCHAT_MIN_SCORE = 75.0

# greetings and sign-offs are short. the score alone is not enough to identify
# them, because rapidfuzz's token_set_ratio ignores word order and duplicated
# tokens: "Are there any courses available?" scored 76.9 against a greeting
# pattern purely on the shared words "are" and "there", which suppressed
# retrieval and left the model answering a real question from nothing.
_CHITCHAT_MAX_WORDS = 4



def _is_conversational(message: str) -> bool:
    """Cheap structural guard applied before the chit-chat classifier.

    Both conditions have to hold: the message must be short, and it must not
    name a subject the chatbot can answer about. Either one alone lets real
    questions through - "fees?" is short but substantive, and a long message
    can still share tokens with a greeting pattern.
    """
    words = re.findall(r"[\w']+", message or "")
    if not words or len(words) > _CHITCHAT_MAX_WORDS:
        return False
    return not conversation.mentions_topic(message)


# a passage is kept only if its dense distance is within this margin of the
# best match for the same query. chosen from measured spreads: for factual
# queries the useful passages sit within ~0.21 of the best, while the noise
# tail on conversational turns starts around +0.37.
_RELEVANCE_MARGIN = 0.30


_UWTSD_HOST = "https://www.uwtsd.ac.uk"
_BYDTERMCYMRU_HOST = "https://termau.cymru"


# ── wellbeing signposting ────────────────────────────────────────────
# required by the project's ethics approval: the form commits to signposting
# external support whenever a participant raises a wellbeing topic, on the
# basis that the study carries a mild risk of distress.
#
# this is appended deterministically rather than left to the model. relying on
# generation would mean the numbers appear only when retrieval happens to
# surface the wellbeing intent AND the model chooses to quote it, which is not
# a safeguard - it was observed answering "wellbeing support please" with no
# helplines at all. a student in distress has to get these every time.
#
# services and numbers are exactly as declared on the ethics form. do not edit
# without re-checking the approved form; they are a condition of approval, not
# a content choice.
#
# plain text, no markdown. the frontend passes replies through linkify() into
# dangerouslySetInnerHTML and does not render markdown, so "**bold**" and "---"
# would appear literally to the student. bullets use a character that renders
# as-is.
_SIGNPOST_EN = (
    "\n\n"
    "If you need to talk to someone now:\n"
    "• NHS 111 Wales — call 111 (option 2)\n"
    "• Samaritans Wales — 116 123 (Welsh language line: 0808 164 0123)\n"
    "• Student Minds — text STUDENT to 85258\n"
    "• Education Support — 08000 562 561\n"
    "• UWTSD Wellbeing — wellbeingsupport@uwtsd.ac.uk"
)

_SIGNPOST_CY = (
    "\n\n"
    "Os oes angen siarad â rhywun nawr:\n"
    "• GIG 111 Cymru — ffoniwch 111 (dewis 2)\n"
    "• Samariaid Cymru — 116 123 (llinell Gymraeg: 0808 164 0123)\n"
    "• Student Minds — tecstiwch STUDENT i 85258\n"
    "• Education Support — 08000 562 561\n"
    "• Lles PCYDDS — wellbeingsupport@uwtsd.ac.uk"
)

# intents that concern a student's wellbeing rather than an administrative
# question. matching any of these triggers the signpost.
_WELLBEING_TAGS = frozenset({
    "wellbeing_general", "wellbeing_crisis", "wellbeing_disability",
    "student_stress",
})

# emotion labels that indicate the student may be struggling, regardless of
# what they asked about. "distressed" is the safeguarding label in
# emotion.py; "stressed" is included because the ethics form covers mild
# distress, not only crisis.
_WELLBEING_EMOTIONS = frozenset({"distressed", "stressed"})

# signposting gets its own pattern rather than reusing conversation's
# "wellbeing" topic signal. that one is tuned for conversational focus and is
# far too narrow here: \bcounsel\b does not match "counselling", and "health",
# "struggling" and "I feel low" all miss it entirely. under-triggering a
# safeguarding feature is the failure that matters, so this is deliberately
# broad, and kept separate so tuning it cannot disturb topic tracking.
_WELLBEING_RE = re.compile(
    r"\b("
    # english
    r"well[-\s]?being|mental\s+health|counsell?\w*|therapy|therapist|"
    r"depress\w*|anxi\w*|lonel\w*|isolated|struggl\w*|"
    r"can'?t\s+cope|cannot\s+cope|not\s+coping|overwhelmed|burn[-\s]?out|"
    r"suicid\w*|self[-\s]?harm|crisis|distress\w*|panic|"
    r"low\s+mood|feel(ing)?\s+low|feel(ing)?\s+down|safeguard\w*|"
    r"my\s+health|health\s+support|help\s+with\s+health|"
    # welsh
    r"lles|iechyd\s+meddwl|cwnsela|gorbryder|iselder|unig|straen|"
    r"ymdopi|hunanladdiad|argyfwng|hunan[-\s]?niweidio"
    r")\b",
    re.I,
)


def _needs_signposting(
    message: str, matched_intent: dict | None, feeling: str,
) -> bool:
    """Whether the reply must carry the external support signpost.

    Deliberately over-inclusive: three independent signals, any of which
    fires. Showing helplines to a student who did not need them costs a few
    lines; withholding them from one who did is the failure mode the ethics
    form exists to prevent.
    """
    if feeling in _WELLBEING_EMOTIONS:
        return True
    if matched_intent and matched_intent.get("tag") in _WELLBEING_TAGS:
        return True
    # keyword pass catches phrasings the intent classifier scores too low on,
    # e.g. "I need help with health"
    return bool(_WELLBEING_RE.search(message or ""))

# how many institutional pages an answer aims to cite. the top-up runs until
# a Welsh answer reaches this many, or the English side is exhausted.
_MIN_INSTITUTIONAL_SOURCES = 3


def _is_institutional(passage: dict) -> bool:
    """True when the passage cites a UWTSD page.

    Only these are offered to the student as sources. The BydTermCymru
    terminology entries are deliberately excluded: they exist so the retriever
    and the model can work with Welsh vocabulary, and a gloss such as
    "cofrestru (Saesneg: register)" tells a student nothing about where the
    answer came from. Citing it would dress a comprehension aid up as
    provenance. The entries remain in retrieval, where they do their real work.
    """
    return str(passage.get("source") or "").startswith(_UWTSD_HOST)


async def _topup_english_sources(
    augmented_query: str, passages: list[dict],
) -> list[dict]:
    """Append English institutional passages to a Welsh result set.

    Welsh passages are preserved and keep their original order; English pages
    are appended only to make up the shortfall, so a Welsh query that already
    found Welsh institutional content is left untouched.
    """
    institutional = [p for p in passages if _is_institutional(p)]
    shortfall = _MIN_INSTITUTIONAL_SOURCES - len(institutional)
    if shortfall <= 0:
        return passages

    try:
        english = await rag.retrieve(augmented_query, top_k=None, lang="en")
    except Exception as e:
        # a failed top-up must never cost the student the Welsh sources
        log.warning("[cy-topup] English retrieval failed: %s", e)
        return passages

    english = _filter_by_relevance(english)
    seen = {p.get("source") for p in passages}
    added: list[dict] = []
    for p in english:
        if len(added) >= shortfall:
            break
        if not _is_institutional(p) or p.get("source") in seen:
            continue
        seen.add(p.get("source"))
        added.append(p)

    if added:
        log.info(
            "[cy-topup] Welsh pass gave %d institutional source(s), appended %d "
            "English page(s)", len(institutional), len(added),
        )
    return passages + added


def _select_sources(passages: list[dict], limit: int = 3) -> list[dict]:
    """Pick the passages to cite: institutional pages, de-duplicated.

    Only UWTSD pages are cited, for the reason given on _is_institutional.
    A Welsh answer can therefore legitimately show fewer than `limit` sources,
    or none, and that is preferable to padding the panel with glossary entries
    that cannot be followed back to anything.

    De-duplication is by URL. Retrieval frequently returns two chunks of the
    same page, which previously rendered as two identical-looking cards.
    """
    chosen: list[dict] = []
    seen: set[str] = set()
    for p in passages:
        if len(chosen) >= limit:
            break
        if not _is_institutional(p):
            continue
        url = str(p.get("source") or "")
        if url in seen:
            continue
        seen.add(url)
        chosen.append(p)
    return chosen


def _category_for(source_url: str | None, lang: str) -> str:
    """Human-readable category badge for a source.

    Derived from the final path segment of the URL. Two cases the previous
    version got wrong: a URL ending in a slash ("https://termau.cymru/") split
    to an empty string and rendered as a blank badge, and the terminology
    service has no meaningful path to derive a label from at all.
    """
    if not source_url:
        return "Gwybodaeth" if lang == "cy" else "Information"
    slug = source_url.rstrip("/").split("/")[-1]
    if not slug or "." in slug:
        return "Gwybodaeth" if lang == "cy" else "Information"
    return slug.replace("-", " ").replace("_", " ").title()


def _filter_by_relevance(
    passages: list[dict], drop_all: bool = False,
) -> list[dict]:
    """Drop retrieved passages that are far worse than the query's best match.

    Passages found by BM25 alone carry no dense distance; they are kept, since
    an exact keyword hit is its own evidence of relevance.

    drop_all short-circuits to an empty list for turns already identified as
    conversational, so the model answers from the dialogue rather than from
    unrelated institutional text.
    """
    if drop_all:
        return []
    scored = [p["score"] for p in passages if p.get("score") is not None]
    if not scored:
        return passages
    cutoff = min(scored) + _RELEVANCE_MARGIN
    return [
        p for p in passages
        if p.get("score") is None or p["score"] <= cutoff
    ]


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest) -> ChatResponse:
    # 1. pick the language. we ALWAYS run the detector on the current
    #    message and let it decide, even when the frontend supplied
    #    runningLang. the frontend hint is stale on the very turn the
    #    user switches language, because its lang state only updates
    #    when a reply lands. trusting runningLang here used to cause
    #    the bot to reply in the previous language for one turn.
    detected = welsh.detect_language(req.message)
    lang = detected or req.runningLang or "en"

    # loud log so we can confirm in the backend window that detection
    # is firing on the current turn and not falling back to a stale
    # frontend hint. the snippet is trimmed for readability.
    log.info(
        "[lang-detect] msg=%r detected=%s runningLang=%s -> lang=%s",
        req.message[:60], detected, req.runningLang, lang,
    )

    # 1a. pre-LLM injection filter. if the message matches a known
    # adversarial pattern (prompt extraction, role confusion, scraping
    # request, etc.) we short-circuit with a hardcoded same-language
    # refusal. saves API tokens and guarantees a safe answer that
    # cannot be drifted by the model.
    injection_label = _is_injection_attempt(req.message, lang)
    if injection_label is not None:
        log.warning(
            "[injection-filter] BLOCKED, lang=%s pattern=%r msg=%r",
            lang, injection_label, req.message[:80],
        )
        refusal = _INJECTION_REFUSAL_CY if lang == "cy" else _INJECTION_REFUSAL_EN
        return ChatResponse(
            reply=refusal,
            lang=lang,
            intent="injection_refusal",
            emotion="neutral",
            sources=[],
        )

    # 2. bilingual query augmentation. a Welsh query gets its terms mapped
    #    to English so the English-indexed vector store still matches.
    #    ref: Jones, D. (2023) BydTermCymru bilingual terminology dataset
    augmented = welsh.augment_query(req.message, lang)

    # 3. resolve pronouns ("is it expensive?" after asking about Computing)
    #    and pull out any student profile info.
    history_dicts = [t.model_dump() for t in req.history]

    # filter history to turns matching the current language. prior turns
    # in the other language confuse the LLM, it weighs them as a lang
    # signal and code-mixes the reply. we keep state-building on the
    # full history (so name/course/year still resolve) but only the
    # same-language turns make it to the LLM prompt.
    history_for_llm = [
        t for t in history_dicts
        if welsh.detect_language(t.get("text") or "") == lang
    ]

    state = conversation.build_state(req.message, history_dicts)

    # 4. light-weight emotion classification so we can soften replies if
    #    the user sounds stressed.
    feeling = emotion.detect(req.message, history_dicts, lang)

    # 5. match against the knowledge base intents. returns None if nothing
    #    scores above the confidence threshold.
    # the history boost is applied only when the message actually refers back
    # to something. it multiplies the previous topic's intents by up to 2.5, so
    # on a fresh question it produced tags like `courses_computing` for "wow",
    # which then reached the model as "LIKELY INTENT: courses_computing" - an
    # explicit instruction to talk about computing courses. the same anaphora
    # test governs topic inheritance in conversation.build_state.
    anaphoric = conversation.is_anaphoric(req.message)
    matched_intent = intent.classify(
        augmented, lang, history_dicts, use_history_boost=anaphoric,
    )

    # 5a. conversational turns are re-classified with the intent history boost
    # disabled, for labelling only. the boost multiplies the previous topic's
    # intents by up to 2.5, so a closing message like "thanks" would otherwise
    # be tagged `courses_general` purely because the previous turn was about
    # courses. the reply itself is still generated by the LLM on every turn -
    # we do not substitute canned text, because the model needs to see the
    # conversation to respond to it in context.
    chitchat = None
    if _is_conversational(req.message):
        chitchat = intent.classify(
            req.message, lang,
            history=None,
            min_score=_CHITCHAT_MIN_SCORE,
            use_history_boost=False,
        )
    is_chitchat = chitchat is not None and chitchat.get("tag") in _CHITCHAT_TAGS
    if is_chitchat:
        matched_intent = chitchat
        # answer these with no history at all. two narrower versions failed,
        # in opposite directions:
        #   - keep the last exchange -> the previous answer is right there and
        #     the model restates it ("thank you very much" came back as the
        #     whole accommodation explanation again)
        #   - keep only the student's turns -> history now reads as a list of
        #     unanswered questions, so the model answers them. worse: a later
        #     "what are the tuition fees?" opened "let me cover both"
        # a thank-you doesn't need the transcript to be answered properly, and
        # dropping it is the only variant that leaves nothing behind to leak
        # into the next turn.
        history_for_llm = []
        log.info(
            "[chitchat] tag=%s score=%.1f, suppressing retrieval and history",
            chitchat["tag"], chitchat.get("score", 0.0),
        )

    # 6. RAG retrieval, pulls top-k grounded passages from Chroma.
    #    when lang is cy the retriever first tries lang-filtered passages
    #    so the LLM has Welsh phrasing to ground in.
    #    ref: Lewis, P. et al. (2020) Retrieval-Augmented Generation for
    #    Knowledge-Intensive NLP Tasks, NeurIPS 2020
    passages = await rag.retrieve(augmented, top_k=None, lang=lang)

    # 6a. relevance gate.
    #
    # retrieval always returns its top-k, however poor the matches are, so a
    # low-signal turn ("thanks", "wow") comes back with whatever happens to sit
    # nearest in embedding space. feeding those to the model is what caused it
    # to answer "Thanks" with a course listing: given UWTSD passages in its
    # prompt, an 8B model will talk about them.
    #
    # we drop passages whose dense distance is far worse than the best match
    # for the same query. an absolute cut-off does not work here - measured
    # best-match distances for substantive queries (0.50-0.83) overlap almost
    # entirely with conversational ones (0.45-0.98) - but the *spread* does
    # separate them, because a real question has several passages clustered
    # near the best while a conversational turn has one loose match and a tail
    # of noise.
    #
    # this is the semantic threshold filtering described by Kong et al. (24),
    # applied relatively rather than absolutely.
    passages = _filter_by_relevance(passages, drop_all=is_chitchat)

    # 6b. English institutional top-up for Welsh answers.
    #
    # UWTSD publishes far less Welsh than English: only 322 of the 2,835
    # indexed corpus chunks are Welsh. A Welsh query is therefore usually
    # grounded in the BydTermCymru terminology layer, and the student is
    # offered three dictionary glosses - "cofrestru (Saesneg: register)" - in
    # place of anything they could actually act on.
    #
    # Where the Welsh pass yields too few institutional pages we run a second
    # retrieval over the English side and append the pages it finds. The
    # English page is the authoritative source in either language; withholding
    # it because of the query language leaves the Welsh speaker worse informed,
    # which is the opposite of what the Welsh Language Act provision intends.
    # The reply itself stays in Welsh - only the citations differ - and each
    # topped-up source carries lang="en" so the interface can label it.
    #
    # augment_query has already mapped the Welsh query's terms into English,
    # so `augmented` is the appropriate probe for the English side.
    #
    # The topped-up set is kept separate from `passages` and is used ONLY for
    # attribution. Feeding English passages into a Welsh prompt would give the
    # model a strong English signal to imitate, which is exactly the drift the
    # language lock exists to prevent, so the generation context stays Welsh.
    source_passages = passages
    if lang == "cy" and not is_chitchat:
        source_passages = await _topup_english_sources(augmented, passages)

    # 7. ask the LLM (Claude by default, Ollama fallback) to write the reply.
    try:
        reply = await llm.generate(
            message=req.message,
            augmented=augmented,
            lang=lang,
            history=history_for_llm,
            passages=passages,
            student_state=state,
            emotion=feeling,
            intent_tag=matched_intent.get("tag") if matched_intent else None,
        )
    except Exception as e:
        # if the LLM is unreachable we return a 503 so the frontend can
        # show a friendly "try again" message instead of crashing.
        log.exception("LLM generation failed")
        raise HTTPException(status_code=503, detail=f"LLM unavailable: {e}")

    # 7a. wellbeing signposting, appended after generation so it cannot be
    # dropped or paraphrased by the model. see _needs_signposting.
    if _needs_signposting(req.message, matched_intent, feeling):
        signpost = _SIGNPOST_CY if lang == "cy" else _SIGNPOST_EN
        # guard against duplication when the model has already quoted a number
        # from the retrieved wellbeing passages
        if "116 123" not in reply:
            reply = reply.rstrip() + signpost
        log.info("[signpost] wellbeing support appended, lang=%s", lang)

    # Extract the three sources shown alongside the reply.
    top_sources = []
    for passage in _select_sources(source_passages, limit=3):
        text = passage.get("text", "")
        excerpt = text[:150].strip() if text else ""

        # Get URL from either 'source' or 'url' field (corpus uses 'url')
        source_url = passage.get("source") or passage.get("url")

        top_sources.append(SourceRef(
            title=passage.get("title", "Information"),
            category=_category_for(source_url, lang),
            excerpt=excerpt,
            language=passage.get("lang", "en"),
            url=source_url if source_url else None,
        ))

    return ChatResponse(
        reply=reply,
        lang=lang,
        intent=(matched_intent.get("tag") if matched_intent else None),
        emotion=feeling,
        sources=top_sources,
    )
