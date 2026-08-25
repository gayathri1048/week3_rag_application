"""Chroma-backed vector store: one collection of ticket chunks."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

from .embeddings import get_embedding_function

if TYPE_CHECKING:  # import only for typing — importing it at runtime would make
    # retrieval depend on ingestion, which already depends on retrieval.
    from ..ingestion.chunker import Chunk

logger = logging.getLogger(__name__)

# Chroma writes upserts in batches; keep them well under the server's payload cap.
UPSERT_BATCH_SIZE = 256


class VectorStore:
    """Thin wrapper over a persistent Chroma collection.

    Cosine distance is requested explicitly so `1 - distance` is a meaningful
    similarity score in the 0..1 range for the UI to display.
    """

    def __init__(self, path: Path, collection_name: str) -> None:
        import chromadb
        from chromadb.config import Settings as ChromaSettings

        path.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.collection_name = collection_name
        self._client = chromadb.PersistentClient(
            path=str(path),
            settings=ChromaSettings(anonymized_telemetry=False, allow_reset=True),
        )
        self._collection = self._get_or_create()

    def _get_or_create(self) -> Any:
        return self._client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=get_embedding_function(),
            metadata={"hnsw:space": "cosine"},
        )

    # ----------------------------------------------------------------- writes
    def upsert(self, chunks: "Sequence[Chunk]") -> int:
        """Insert or replace chunks by id. Returns the number written."""
        if not chunks:
            return 0

        for start in range(0, len(chunks), UPSERT_BATCH_SIZE):
            batch = chunks[start : start + UPSERT_BATCH_SIZE]
            self._collection.upsert(
                ids=[c.chunk_id for c in batch],
                documents=[c.text for c in batch],
                metadatas=[c.metadata for c in batch],
            )
            logger.debug("upserted %d chunks (offset %d)", len(batch), start)

        return len(chunks)

    def delete_ticket(self, ticket_id: str) -> None:
        """Remove every chunk belonging to one ticket."""
        self._collection.delete(where={"ticket_id": ticket_id})

    def reset(self) -> None:
        """Drop and recreate the collection — used by `ingest(reset=True)`."""
        logger.warning("resetting collection %s", self.collection_name)
        try:
            self._client.delete_collection(self.collection_name)
        except Exception as exc:  # Chroma raises a NotFoundError subclass when absent
            logger.debug("delete_collection no-op: %s", exc)
        self._collection = self._get_or_create()

    # ------------------------------------------------------------------ reads
    def query(
        self,
        text: str,
        top_k: int,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Nearest-neighbour search. Returns raw hits with `score` in 0..1."""
        result = self._collection.query(
            query_texts=[text],
            n_results=top_k,
            where=where or None,
            include=["documents", "metadatas", "distances"],
        )

        # Chroma returns one list per query text; we only ever send one.
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        hits: list[dict[str, Any]] = []
        for chunk_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
            hits.append(
                {
                    "chunk_id": chunk_id,
                    "text": document or "",
                    "metadata": metadata or {},
                    # Cosine distance is 0..2; clamp so a score never goes negative.
                    "score": max(0.0, min(1.0, 1.0 - float(distance))),
                }
            )
        return hits

    def get_all(self, limit: int | None = None, offset: int = 0) -> list[dict[str, Any]]:
        """Page through stored chunks — backs the ticket list and stats endpoints."""
        result = self._collection.get(
            limit=limit,
            offset=offset or None,
            include=["documents", "metadatas"],
        )
        ids = result.get("ids") or []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        return [
            {"chunk_id": i, "text": d or "", "metadata": m or {}}
            for i, d, m in zip(ids, documents, metadatas)
        ]

    def count(self) -> int:
        return self._collection.count()
