"""Application settings, loaded from environment / .env."""

from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

BACKEND_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=BACKEND_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM Provider ---
    # Default is 'ollama' for 100% free local AI without consuming API tokens
    llm_provider: Literal["ollama", "anthropic", "fallback"] = "ollama"

    # --- Ollama (100% Free Local AI) ---
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2"

    # --- Anthropic ---
    anthropic_api_key: str | None = None
    claude_model: str = "claude-opus-5"
    claude_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    claude_max_tokens: int = 16_000

    # --- Retrieval ---
    chroma_path: Path = BACKEND_ROOT / "chroma_db"
    collection_name: str = "support_tickets"
    top_k: int = 6
    min_relevance: float = Field(default=0.25, ge=0.0, le=1.0)

    # --- Ingestion ---
    documents_dir: Path = BACKEND_ROOT / "documents"
    chunk_size: int = 900
    chunk_overlap: int = 150

    # --- Server ---
    # Backend-only deployment: allow any origin so external API testers
    # (Swagger UI on a LAN IP, Postman, another machine's browser) aren't blocked.
    #
    # NoDecode is load-bearing: pydantic-settings JSON-decodes complex fields in
    # the .env source itself, *before* field validators run, so a comma-separated
    # CORS_ORIGINS raises SettingsError at import time rather than reaching
    # `_split_origins` below. NoDecode hands the raw string to the validator.
    cors_origins: Annotated[list[str], NoDecode] = ["*"]
    log_level: str = "info"

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _split_origins(cls, v: object) -> object:
        """Accept a comma-separated string from .env as well as a real list."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    @field_validator("chroma_path", "documents_dir", mode="after")
    @classmethod
    def _absolutise(cls, v: Path) -> Path:
        return v if v.is_absolute() else (BACKEND_ROOT / v).resolve()


@lru_cache
def get_settings() -> Settings:
    return Settings()
