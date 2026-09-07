"""Langfuse observability client for customer support RAG (Langfuse SDK v4 compatible).

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
    host_url = settings.langfuse_base_url or settings.langfuse_host
    try:
        from langfuse import Langfuse

        _langfuse_instance = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=host_url,
        )
        logger.info("Langfuse observability client initialized successfully (host: %s)", host_url)
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

        # 1. Start root RAG chain span
        root_span = client.start_observation(
            name="customer_support_rag",
            as_type="chain",
            input={"question": query},
            metadata={
                "trace_id": trace_id,
                "prompt_version": prompt_version,
                "retrieved_chunk_count": len(chunks_payload),
                "tags": tags or ["rag", "customer-support-rag", "week5-eval"],
                **(metadata or {}),
            },
        )

        # 2. Add retrieval span
        retriever_span = root_span.start_observation(
            name="knowledge_retrieval",
            as_type="retriever",
            input={"query": query},
            output={"chunks": chunks_payload},
            metadata={"top_k": len(chunks_payload)},
        )
        retriever_span.end()

        # 3. Add generation span
        gen_span = root_span.start_observation(
            name="rag_generation",
            as_type="generation",
            model=model,
            model_parameters=model_params or {"temperature": 0.0},
            input=[
                {"role": "system", "content": f"UBP Support Assistant ({prompt_version})"},
                {"role": "user", "content": query},
            ],
            output=answer,
            metadata={"latency_ms": latency_ms},
        )
        gen_span.end()

        # Complete root observation
        root_span.update(output={"answer": answer})
        root_span.end()
        client.flush()

        logger.info(
            "📡 [Langfuse] Logged live RAG trace: query='%s...', chunks=%d, model=%s",
            query[:40].replace("\n", " "),
            len(chunks_payload),
            model,
        )

        return root_span
    except Exception as exc:
        logger.warning("Could not push trace %s to Langfuse: %s", trace_id, exc)
        return None
