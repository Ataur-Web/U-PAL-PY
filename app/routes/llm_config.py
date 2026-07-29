"""GET/POST /api/llm-config, runtime LLM switching.

lets us flip between Claude and Ollama (and swap the specific model) without
restarting the backend. state lives in-memory and resets on restart, which
is fine because this is a dev/demo tool.

security note: this endpoint has no auth. we only expose it through the
ngrok tunnel for the operator. if we ever expose it publicly it needs
Basic Auth or similar.
"""
from __future__ import annotations

import logging
from typing import Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.config import get_settings
from app.services import llm


router = APIRouter()
log = logging.getLogger("u-pal-py.llm_config")


# hard-coded model lists. we surface these in the frontend dropdown so the
# operator can pick one without typing. keeping them here means we don't
# have to call the provider's list-models endpoint on every request.
_ANTHROPIC_MODELS = [
    {"id": "claude-haiku-4-5-20251001", "label": "Claude Haiku 4.5, cheapest (recommended)"},
    {"id": "claude-3-5-sonnet-latest", "label": "Claude 3.5 Sonnet, balanced"},
    {"id": "claude-sonnet-4-5",        "label": "Claude Sonnet 4.5, newer balanced"},
    {"id": "claude-opus-4-1",          "label": "Claude Opus 4.1, premium reasoning"},
    {"id": "claude-3-opus-latest",     "label": "Claude 3 Opus, legacy premium"},
]

_OLLAMA_MODELS = [
    {"id": "llama3.1:8b-instruct-q5_K_M", "label": "Llama 3.1 8B Instruct Q5_K_M"},
    {"id": "llama3.1:8b",                 "label": "Llama 3.1 8B"},
    {"id": "llama3.2:3b",                 "label": "Llama 3.2 3B"},
    {"id": "qwen2.5:7b",                  "label": "Qwen 2.5 7B"},
    {"id": "mistral:7b",                  "label": "Mistral 7B"},
]


class LLMConfigResponse(BaseModel):
    # two providers: Claude (cloud, primary) and Ollama (local fallback)
    provider:             Literal["anthropic", "ollama"]
    anthropic_model:      str
    ollama_model:         str
    anthropic_available:  bool
    ollama_available:     bool
    anthropic_models:     list[dict]
    ollama_models:        list[dict]


class LLMConfigRequest(BaseModel):
    provider: Literal["anthropic", "ollama"] | None = None
    model:    str | None                            = Field(default=None, max_length=120)


def _snapshot() -> LLMConfigResponse:
    # returns the current config so GET and POST can both reuse the shape
    s = get_settings()
    return LLMConfigResponse(
        provider            = llm.get_active_provider() if (s.anthropic_api_key or s.ollama_url) else "ollama",
        anthropic_model     = s.anthropic_model,
        ollama_model        = s.ollama_model,
        anthropic_available = bool(s.anthropic_api_key),
        ollama_available    = bool(s.ollama_url),
        anthropic_models    = _ANTHROPIC_MODELS,
        ollama_models       = _OLLAMA_MODELS,
    )


@router.get("/llm-config", response_model=LLMConfigResponse)
async def get_config() -> LLMConfigResponse:
    return _snapshot()


@router.post("/llm-config", response_model=LLMConfigResponse)
async def update_config(req: LLMConfigRequest) -> LLMConfigResponse:
    s = get_settings()

    if req.provider is None and req.model is None:
        raise HTTPException(400, "must provide provider and/or model")

    # reject the switch early if the target provider has no key/url configured
    if req.provider == "anthropic" and not s.anthropic_api_key:
        raise HTTPException(400, "Claude is not configured (ANTHROPIC_API_KEY missing).")
    if req.provider == "ollama" and not s.ollama_url:
        raise HTTPException(400, "Ollama is not configured (OLLAMA_URL missing).")

    # we mutate the cached Settings object in place. this works because
    # get_settings() is lru_cache-wrapped, so every caller shares the same
    # instance.
    # ref: https://docs.python.org/3/library/functools.html#functools.lru_cache
    if req.provider is not None:
        s.llm_provider = req.provider
    if req.model is not None:
        # apply the new model ID to whichever provider we just selected
        # (or the current one, if the POST only changed the model).
        target = req.provider or s.llm_provider
        if target == "anthropic":
            s.anthropic_model = req.model
        else:
            s.ollama_model = req.model

    # drop the cached LLM client so the next /api/chat request rebuilds
    # it with the new settings.
    llm.reload_client()
    log.info(
        "LLM config updated: provider=%s anthropic=%s ollama=%s",
        s.llm_provider, s.anthropic_model, s.ollama_model,
    )
    return _snapshot()
