"""Langfuse observability client for customer support RAG.

Provides thread-safe, non-blocking telemetry logging for user queries,
retrieval spans, and LLM generations.
"""

from __future__ import annotations

import logging
from typing import Any

from ..config import get_settings

logger = logging.getLogger(__name__)

_langfuse_instance: Any = None


def is_langfuse_configured() -> bool:
    """Check if valid Langfuse credentials are present in settings."""
    settings = get_settings()
    if not settings.langfuse_enabled:
        return False
    return bool(settings.langfuse_public_key and settings.langfuse_secret_key)


def get_langfuse_client() -> Any | None:
    """Initialize and return singleton Langfuse client instance if configured."""
    global _langfuse_instance
    if _langfuse_instance is not None:
        return _langfuse_instance

    if not is_langfuse_configured():
        return None

    settings = get_settings()
    try:
        from langfuse import Langfuse

        _langfuse_instance = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
        logger.info("Langfuse observability client initialized successfully (host: %s)", settings.langfuse_host)
        return _langfuse_instance
    except Exception as exc:
        logger.warning("Failed to initialize Langfuse client: %s", exc)
        return None


def log_rag_trace(
    *,
    trace_id: str | None = None,
    query: str,
    retrieved_chunks: list[dict[str, Any]] | list[Any],
    answer: str,
    model: str = "claude-3-5-sonnet-20241022",
    model_params: dict[str, Any] | None = None,
    prompt_version: str = "v2.4-support-rag",
    latency_ms: float | None = None,
    tags: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Any | None:
    """Log a complete RAG interaction to Langfuse (trace, retrieval span, generation).

    Gracefully no-ops if Langfuse is not configured or fails.
    """
    client = get_langfuse_client()
    if client is None:
        return None

    try:
        # Prepare structured chunks representation
        chunks_payload = []
        for c in retrieved_chunks:
            if hasattr(c, "model_dump"):
                c_dict = c.model_dump()
            elif isinstance(c, dict):
                c_dict = c
            else:
                c_dict = {"text": str(c)}
            chunks_payload.append(c_dict)

        # 1. Create main Langfuse trace
        trace = client.trace(
            id=trace_id,
            name="customer_support_rag",
            input={"question": query},
            output={"answer": answer},
            metadata={
                "prompt_version": prompt_version,
                "retrieved_chunk_count": len(chunks_payload),
                **(metadata or {}),
            },
            tags=tags or ["rag", "week5-eval"],
        )

        # 2. Add retrieval span
        trace.span(
            name="knowledge_retrieval",
            input={"query": query},
            output={"chunks": chunks_payload},
            metadata={"top_k": len(chunks_payload)},
        )

        # 3. Add generation span
        trace.generation(
            name="rag_generation",
            model=model,
            model_parameters=model_params or {"temperature": 0.0},
            input=[
                {"role": "system", "content": f"UBP Support Migration Assistant ({prompt_version})"},
                {"role": "user", "content": query},
            ],
            output=answer,
            metadata={"latency_ms": latency_ms},
        )

        return trace
    except Exception as exc:
        logger.warning("Could not push trace %s to Langfuse: %s", trace_id, exc)
        return None
