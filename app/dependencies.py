"""FastAPI dependency providers.

The vector store and retriever are process-wide singletons: Chroma holds an open
handle to its persistence directory and the embedding model costs a few hundred ms
to load, so neither should be rebuilt per request.
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import Depends

from .config import Settings, get_settings
from .generation.answerer import Answerer
from .retrieval.retriever import Retriever
from .retrieval.vector_store import VectorStore


@lru_cache
def get_vector_store() -> VectorStore:
    settings = get_settings()
    return VectorStore(settings.chroma_path, settings.collection_name)


@lru_cache
def get_retriever() -> Retriever:
    settings = get_settings()
    return Retriever(
        store=get_vector_store(),
        top_k=settings.top_k,
        min_relevance=settings.min_relevance,
    )


def get_answerer() -> Answerer:
    """Not cached — constructing it is cheap and it holds no connection state."""
    return Answerer(retriever=get_retriever())


SettingsDep = Depends(get_settings)
StoreDep = Depends(get_vector_store)
RetrieverDep = Depends(get_retriever)
AnswererDep = Depends(get_answerer)

__all__ = [
    "Settings",
    "get_settings",
    "get_vector_store",
    "get_retriever",
    "get_answerer",
    "SettingsDep",
    "StoreDep",
    "RetrieverDep",
    "AnswererDep",
]
