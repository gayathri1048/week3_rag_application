"""Embedding function for the vector store.

Anthropic does not ship an embeddings endpoint, so retrieval uses a local model:
Chroma's bundled `all-MiniLM-L6-v2` (384-dim, ONNX runtime, no API key, no network
call after the first download). That keeps the RAG loop free and offline — Claude is
used only for generation.

To swap in a hosted embedder, replace `get_embedding_function()` with one of
Chroma's provider functions (e.g. `VoyageAIEmbeddingFunction`) and re-run the
ingest — the collection must be rebuilt because vector dimensions change.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any

logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


@lru_cache
def get_embedding_function() -> Any:
    """Return a cached Chroma embedding function.

    Cached because loading the ONNX model takes a few hundred ms and it is
    thread-safe to share across requests.
    """
    from chromadb.utils import embedding_functions

    logger.info("loading embedding model %s (%d-dim)", EMBEDDING_MODEL, EMBEDDING_DIM)
    return embedding_functions.DefaultEmbeddingFunction()
