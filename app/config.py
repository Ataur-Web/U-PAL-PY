"""central settings, loaded once from .env on first call."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "app" / "data"


# we load .env with override=True so values in the file win over anything
# already set in the OS environment. some shells inject an empty
# ANTHROPIC_API_KEY for safety which would otherwise hide the real key
# we put in .env.
# ref: https://pypi.org/project/python-dotenv/
load_dotenv(PROJECT_ROOT / ".env", override=True)


class Settings(BaseSettings):
    # this tells pydantic-settings where to look and to ignore unknown keys
    # ref: https://docs.pydantic.dev/latest/concepts/pydantic_settings/
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # which LLM to use at boot. can still be swapped at runtime via /api/llm-config
    #   "anthropic" = Claude (cloud, default)
    #   "ollama"    = local fallback for offline demos
    llm_provider:           str  = Field(default="anthropic")

    # Anthropic Claude settings. haiku is the cheapest option and it handles
    # Welsh fine, so we picked it as the default.
    # ref: https://docs.claude.com/en/docs/about-claude/models
    anthropic_api_key:         str  = Field(default="")
    anthropic_model:           str  = Field(default="claude-haiku-4-5-20251001")
    anthropic_temperature:     float = Field(default=0.4)
    anthropic_timeout_seconds: int  = Field(default=60)
    anthropic_max_tokens:      int  = Field(default=1024)

    # Ollama (local fallback)
    ollama_url:             str  = Field(default="http://localhost:11434")
    ollama_model:           str  = Field(default="llama3.1:8b-instruct-q5_K_M")
    ollama_num_ctx:         int  = Field(default=4096)
    ollama_timeout_seconds: int  = Field(default=60)

    # Chroma vector store (for RAG retrieval)
    # ref: https://docs.trychroma.com/getting-started
    chroma_persist_dir:     str  = Field(default="./chroma_db")
    chroma_collection:      str  = Field(default="uwtsd_corpus")
    embedding_model:        str  = Field(
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    )

    # server
    port:       int = Field(default=3001)
    log_level:  str = Field(default="info")

    # how many passages we pull from the vector store per question,
    # and the char cap per passage before we send them to the LLM
    rag_top_k:      int = Field(default=2)
    rag_max_chars:  int = Field(default=500)


# we cache the Settings object so .env is only parsed once per process.
# ref: https://docs.python.org/3/library/functools.html#functools.lru_cache
@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
