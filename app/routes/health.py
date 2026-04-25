"""GET /api/health, live connectivity check for Claude + Ollama + Chroma + Welsh.

the Next.js frontend proxies this endpoint to fill in the CONNECTION card.
"""
from __future__ import annotations

import asyncio
import logging
import subprocess
from functools import lru_cache
from pathlib import Path

import httpx
from fastapi import APIRouter

from app.config import PROJECT_ROOT, get_settings
from app.services import llm, rag, welsh


@lru_cache(maxsize=1)
def _build_sha() -> str:
    # short git sha so the operator can confirm the running backend is
    # on the expected commit. read once at startup, cached forever.
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(PROJECT_ROOT),
            stderr=subprocess.DEVNULL,
            timeout=2.0,
        )
        return out.decode("ascii").strip() or "unknown"
    except Exception:
        return "unknown"


router = APIRouter()
log = logging.getLogger("u-pal-py.health")

# these headers help when we're tunnelling through ngrok (skips the
# browser-warning interstitial) and give the request a useful user-agent.
# ref: https://ngrok.com/docs/cloud-edge/modules/browser-warning/
CF_HEADERS = {
    "User-Agent":                 "UPal-UWTSD-Chatbot/1.0",
    "ngrok-skip-browser-warning": "true",
}


async def _probe_ollama(base_url: str) -> str:
    # quick ping against /api/tags. if it 200s, Ollama is alive.
    if not base_url:
        return "not_configured"
    try:
        async with httpx.AsyncClient(timeout=6.0) as client:
            r = await client.get(f"{base_url}/api/tags", headers=CF_HEADERS)
            return "connected" if r.is_success else f"http_{r.status_code}"
    except httpx.TimeoutException:
        return "timeout"
    except Exception:
        return "offline"


async def _probe_anthropic(api_key: str) -> str:
    # we hit GET /v1/models rather than doing a generation call, so the
    # health check doesn't burn tokens every minute.
    # ref: https://docs.claude.com/en/api/models-list
    if not api_key:
        return "not_configured"
    url = "https://api.anthropic.com/v1/models"
    headers = {
        "x-api-key":         api_key,
        "anthropic-version": "2023-06-01",
    }
    try:
        async with httpx.AsyncClient(timeout=6.0) as client:
            r = await client.get(url, headers=headers)
            if r.is_success:
                return "connected"
            if r.status_code in (401, 403):
                return "auth_error"
            return f"http_{r.status_code}"
    except httpx.TimeoutException:
        return "timeout"
    except Exception:
        return "offline"


async def _probe_chroma() -> tuple[str, int]:
    # this is a sync call under the hood so we push it to a worker thread
    # with asyncio.to_thread, otherwise we'd block the event loop.
    try:
        count = await asyncio.to_thread(rag.get_collection_count)
        return ("ready" if count > 0 else "empty"), count
    except Exception as e:
        log.warning("Chroma probe failed: %s", e)
        return "offline", 0


@router.get("/health")
async def health() -> dict:
    s = get_settings()

    # asyncio.gather runs all three probes in parallel instead of one after
    # the other, so the health check returns in ~6 s worst case rather than
    # ~18 s if all three timed out.
    # ref: https://docs.python.org/3/library/asyncio-task.html#asyncio.gather
    anthropic_status, ollama_status, (chroma_status, chroma_count) = await asyncio.gather(
        _probe_anthropic(s.anthropic_api_key),
        _probe_ollama(s.ollama_url),
        _probe_chroma(),
    )

    # figure out which provider the llm router will actually use, so the UI
    # can mark the right row as ACTIVE.
    provider_pref = s.llm_provider.lower()
    if provider_pref == "anthropic" and s.anthropic_api_key:
        active_provider = "anthropic"
    else:
        active_provider = "ollama"

    return {
        "status":          "OK",
        "buildSha":        _build_sha(),
        # generation providers
        "provider":        active_provider,
        "anthropic":       anthropic_status,
        "anthropicModel":  s.anthropic_model if s.anthropic_api_key else None,
        "ollama":          ollama_status,
        "ollamaModel":     s.ollama_model or None,
        # RAG vector store
        "chroma":          chroma_status,
        "chromaDocs":      chroma_count,
        # Welsh detector
        "welsh":           "active" if welsh.is_loaded() else "not_loaded",
        "bilingualTerms":  welsh.bilingual_map_size(),
        "welshVocab":      welsh.vocab_size(),
    }
