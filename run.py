"""entrypoint: `python run.py` to start the backend.

we have this file because `python -m uvicorn app.main:app` segfaults on
Windows inside the sentence-transformers model load. uvicorn's CLI imports
a stack of asyncio and signal machinery BEFORE app.main, which interacts
badly with torch's native DLL init.

importing app.main first (in a clean Python process) and only then
importing uvicorn avoids the crash.
ref: https://www.uvicorn.org/#usage (programmatic run())
"""
from __future__ import annotations

# this must come first, it triggers the RAG / HuggingFace / torch init
# in a clean Python context before uvicorn touches anything.
from app.main import app  # noqa: E402

import uvicorn  # noqa: E402


if __name__ == "__main__":
    # host 0.0.0.0 so the ngrok tunnel can reach it from outside localhost
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=3001,
        log_level="info",
    )
