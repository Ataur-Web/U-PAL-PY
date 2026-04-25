"""POST /api/translate, translate a bot response into the opposite language.

the frontend calls this lazily when the user clicks the translate button on
a message. we reuse the same LLM client as /api/chat so no extra model is
pulled in.

request:
    { "text": "...", "from_lang": "cy", "to_lang": "en" }

response:
    { "translation": "...", "from_lang": "cy", "to_lang": "en" }
"""
from __future__ import annotations

import logging
from typing import Literal

from fastapi import APIRouter, HTTPException
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field

# we wrap the optional Anthropic import in try/except because the package
# is an optional dep. without it, we fall back to Ollama.
# ref: https://python.langchain.com/docs/integrations/chat/anthropic/
try:
    from langchain_anthropic import ChatAnthropic
except Exception:
    ChatAnthropic = None

from app.config import get_settings

router   = APIRouter()
log      = logging.getLogger("u-pal-py.translate")
# we cache one client between requests instead of rebuilding it every call
_client: BaseChatModel | None = None


def _get_client() -> BaseChatModel:
    # this picks the LLM based on LLM_PROVIDER in .env. temperature is low
    # because we want a faithful translation, not a creative one.
    # ref: Jurafsky, D. and Martin, J.H. (2024) Speech and Language
    # Processing, 3rd ed., chapter on MT evaluation
    global _client
    if _client is not None:
        return _client

    s = get_settings()
    provider = (s.llm_provider or "anthropic").lower().strip()

    if provider == "anthropic" and ChatAnthropic is not None and s.anthropic_api_key:
        _client = ChatAnthropic(
            model=s.anthropic_model,
            api_key=s.anthropic_api_key,
            temperature=0.1,
            timeout=s.anthropic_timeout_seconds,
            max_tokens=s.anthropic_max_tokens,
        )
        return _client

    # Ollama fallback for offline demos.
    # the ngrok-skip-browser-warning header is needed when the Ollama URL
    # points to an ngrok free-tier tunnel.
    _client = ChatOllama(
        model=s.ollama_model,
        base_url=s.ollama_url,
        num_ctx=2048,
        temperature=0.1,
        timeout=s.ollama_timeout_seconds,
        client_kwargs={"headers": {"ngrok-skip-browser-warning": "true"}},
    )
    return _client


# we keep two system prompts rather than one dynamic prompt, it's easier to
# read and to tweak each direction independently.
_SYSTEM_EN_TO_CY = (
    "You are a professional Welsh translator. "
    "Translate the text below from English into natural, fluent Welsh. "
    "Preserve the tone, formality level, and meaning exactly. "
    "Return ONLY the translated Welsh text, no explanations, no brackets, "
    "no English words unless they are proper nouns or technical terms with no Welsh equivalent."
)

_SYSTEM_CY_TO_EN = (
    "You are a professional Welsh-to-English translator. "
    "Translate the text below from Welsh into natural, fluent British English. "
    "Preserve the tone, formality level, and meaning exactly. "
    "Return ONLY the translated English text, no explanations, no brackets, "
    "no Welsh words unless they are proper nouns or brand names."
)


class TranslateRequest(BaseModel):
    text:      str                    = Field(..., min_length=1, max_length=8000)
    from_lang: Literal["en", "cy"]
    to_lang:   Literal["en", "cy"]


class TranslateResponse(BaseModel):
    translation: str
    from_lang:   Literal["en", "cy"]
    to_lang:     Literal["en", "cy"]


@router.post("/translate", response_model=TranslateResponse)
async def translate(req: TranslateRequest) -> TranslateResponse:
    # no-op when both languages match. saves an API call.
    if req.from_lang == req.to_lang:
        return TranslateResponse(
            translation=req.text,
            from_lang=req.from_lang,
            to_lang=req.to_lang,
        )

    system_prompt = _SYSTEM_CY_TO_EN if req.from_lang == "cy" else _SYSTEM_EN_TO_CY

    client = _get_client()
    try:
        # ainvoke is the async version of invoke. we use it because FastAPI
        # is async and we don't want to block the worker thread.
        # ref: https://python.langchain.com/docs/concepts/runnables/
        result = await client.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=req.text),
        ])
        translation = getattr(result, "content", str(result)).strip()
    except Exception as e:
        log.exception("Translation failed")
        raise HTTPException(status_code=503, detail=f"Translation unavailable: {e}")

    return TranslateResponse(
        translation=translation,
        from_lang=req.from_lang,
        to_lang=req.to_lang,
    )
