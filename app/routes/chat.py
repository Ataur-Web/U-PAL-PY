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
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services import conversation, emotion, intent, llm, rag, welsh


router = APIRouter()
log = logging.getLogger("u-pal-py.chat")


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
