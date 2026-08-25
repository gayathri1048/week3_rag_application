"""Liveness and configuration introspection."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends

from ... import __version__
from ...config import get_settings
from ...dependencies import get_vector_store
from ...generation.answerer import active_model_name, resolve_provider
from ...generation.claude_client import is_configured as is_anthropic_configured
from ...generation.ollama_client import is_ollama_available
from ...retrieval.vector_store import VectorStore
from ...schemas import HealthResponse

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health(store: VectorStore = Depends(get_vector_store)) -> HealthResponse:
    """Report whether the app can actually serve requests, and on which backend."""
    settings = get_settings()
    chunk_count = 0
    vector_ok = True
    try:
        chunk_count = store.count()
    except Exception:
        logger.exception("vector store unreachable")
        vector_ok = False

    ollama_ok = is_ollama_available()
    provider = resolve_provider()

    # Retrieval is the hard requirement: the fallback generation path needs no
    # model at all, so a missing key or a stopped Ollama is not "degraded".
    healthy = vector_ok and chunk_count > 0

    if provider == "fallback" and settings.llm_provider != "fallback":
        logger.warning(
            "LLM_PROVIDER=%s but no backend is reachable — serving template answers",
            settings.llm_provider,
        )

    return HealthResponse(
        status="ok" if healthy else "degraded",
        version=__version__,
        vector_store=vector_ok,
        anthropic_configured=is_anthropic_configured(),
        chunk_count=chunk_count,
        llm_provider=provider,
        ollama_available=ollama_ok,
        model=active_model_name(),
    )
