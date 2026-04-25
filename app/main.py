"""FastAPI entrypoint for the U-Pal backend.

run locally with `python run.py` or `uvicorn app.main:app --port 3001`.
"""
from __future__ import annotations

import logging

from app.config import get_settings

settings = get_settings()
logging.basicConfig(level=settings.log_level.upper())
log = logging.getLogger("u-pal-py")


# DO NOT REORDER THE IMPORTS BELOW.
#
# The RAG stack (sentence-transformers, datasets, pyarrow, torch) loads
# native DLLs that segfault on Windows when another set of native DLLs
# (the anthropic / google / openai transport libs pulled in by the route
# modules) has already been mapped into the process.
#
# importing app.services.rag FIRST, and warming it up, makes sure pyarrow's
# DLLs are loaded before anything else. the route modules can then import
# their LLM SDKs without the crash. reversing this order gives exit code 139.
# ref: https://github.com/pytorch/pytorch/issues/42085 (similar DLL ordering issue)
try:
    from app.services import rag as _rag_warmup
    _warmup_count = _rag_warmup.get_collection_count()
    log.info("RAG warmup (import-time) complete, Chroma has %d docs", _warmup_count)
except Exception:
    log.exception("RAG warmup (import-time) failed, continuing")


import asyncio  # noqa: E402
import os       # noqa: E402

from fastapi import FastAPI                          # noqa: E402
from fastapi.middleware.cors import CORSMiddleware   # noqa: E402

from app.routes import chat, health, llm_config, translate  # noqa: E402


# ref: https://fastapi.tiangolo.com/tutorial/first-steps/
app = FastAPI(
    title="U-PAL-PY",
    version="0.1.0",
    description="Python port of the U-Pal UWTSD chatbot backend.",
)

# CORS setup. the Next.js frontend on Vercel calls this backend through an
# ngrok tunnel, so the origin header is the Vercel domain. we allow all
# origins in dev. in production set ALLOWED_ORIGINS to a comma-separated list.
# ref: https://fastapi.tiangolo.com/tutorial/cors/
_origins_env = os.getenv("ALLOWED_ORIGINS", "*").strip()
_origins = [o.strip() for o in _origins_env.split(",") if o.strip()] or ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# mount all four api routes under /api
app.include_router(health.router,      prefix="/api")
app.include_router(chat.router,        prefix="/api")
app.include_router(translate.router,   prefix="/api")
app.include_router(llm_config.router,  prefix="/api")


async def _maybe_auto_ingest() -> None:
    """if the Chroma collection is empty and AUTO_INGEST=1, build it.

    we use this on Railway so the first deploy builds the index without
    anyone needing to shell in and run `python -m scripts.ingest`.
    """
    if os.getenv("AUTO_INGEST", "").lower() not in ("1", "true", "yes"):
        return

    try:
        from app.services import rag
        from scripts import ingest as ingest_script

        # asyncio.to_thread lets us call a blocking function without
        # freezing the event loop, so health checks still work.
        # ref: https://docs.python.org/3/library/asyncio-task.html#asyncio.to_thread
        count = await asyncio.to_thread(rag.get_collection_count)
        if count > 0:
            log.info("[auto-ingest] collection already has %d docs, skipping", count)
            return

        log.info("[auto-ingest] empty collection, building index on startup ...")
        await asyncio.to_thread(ingest_script.main)
        log.info("[auto-ingest] done")
    except Exception:
        log.exception("[auto-ingest] failed, continuing without corpus")


# ref: https://fastapi.tiangolo.com/advanced/events/
@app.on_event("startup")
async def _startup() -> None:
    log.info(
        "U-PAL-PY starting, provider=%s anthropic=%s ollama=%s@%s",
        settings.llm_provider,
        settings.anthropic_model if settings.anthropic_api_key else "not_configured",
        settings.ollama_model,
        settings.ollama_url,
    )

    # fire and forget. we don't want to block the health probe while the
    # ingest job runs, so we kick it off as a background task.
    asyncio.create_task(_maybe_auto_ingest())


@app.get("/")
async def root() -> dict:
    return {"service": "U-PAL-PY", "version": "0.1.0", "docs": "/docs"}
