"""POST /api/chat, the main chatbot endpoint.

request body:
    {
      "message":     "What are the entry requirements?",
      "runningLang": "en",            # optional, we auto-detect if omitted
      "history":     [                # optional, last N turns
        {"role": "user", "text": "I want to study Computing"},
        {"role": "assistant", "text": "UWTSD offers BSc Computing."}
      ]
    }

response:
    {
      "reply":       "...",
      "lang":        "en" | "cy",
      "intent":      "courses_computing",
      "emotion":     "neutral",
      "sources":     [ { "title": "...", "chars": 450 }, ... ]
    }
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
    title: str
    chars: int


class ChatResponse(BaseModel):
    reply:   str
    lang:    Literal["en", "cy"]
    intent:  str | None  = None
    emotion: str         = "neutral"
    sources: list[SourceRef] = Field(default_factory=list)


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
    matched_intent = intent.classify(augmented, lang, history_dicts)

    # 6. RAG retrieval, pulls top-k grounded passages from Chroma.
    #    when lang is cy the retriever first tries lang-filtered passages
    #    so the LLM has Welsh phrasing to ground in.
    #    ref: Lewis, P. et al. (2020) Retrieval-Augmented Generation for
    #    Knowledge-Intensive NLP Tasks, NeurIPS 2020
    passages = await rag.retrieve(augmented, top_k=None, lang=lang)

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

    return ChatResponse(
        reply=reply,
        lang=lang,
        intent=(matched_intent.get("tag") if matched_intent else None),
        emotion=feeling,
        sources=[SourceRef(title=p.get("title", "corpus"),
                            chars=len(p.get("text", "")))
                 for p in passages],
    )
