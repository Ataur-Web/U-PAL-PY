"""U-PAL-PY, the Python backend for the U-Pal UWTSD bilingual chatbot.

this package wires together the FastAPI routes, the ChromaDB + BM25
hybrid retriever, the Welsh language detector, and the Anthropic /
Ollama LLM clients. the entry point is `app.main:app`.
"""
__version__ = "0.1.0"
