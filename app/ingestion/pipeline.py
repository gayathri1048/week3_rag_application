"""Ingest orchestration: documents/ -> chunks -> vector store.

Run standalone to (re)build the index:

    python -m app.ingestion.pipeline           # incremental upsert
    python -m app.ingestion.pipeline --reset    # drop the collection first
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from ..config import get_settings
from ..retrieval.vector_store import VectorStore
from ..schemas import IngestStats
from .chunker import Chunk, chunk_ticket
from .loaders import load_tickets

logger = logging.getLogger(__name__)


def ingest(
    store: VectorStore,
    documents_dir: Path | None = None,
    reset: bool = False,
) -> IngestStats:
    """Read every ticket file, chunk it, and upsert into the vector store.

    Upsert is keyed on `{ticket_id}::{index}`, so re-running after editing a ticket
    replaces its chunks in place. `reset=True` is needed only when chunk boundaries
    shrink (fewer chunks than before would otherwise leave orphans behind).
    """
    settings = get_settings()
    source_dir = documents_dir or settings.documents_dir
    started = time.perf_counter()

    if reset:
        store.reset()

    tickets, files_read = load_tickets(source_dir)

    chunks: list[Chunk] = []
    for ticket in tickets:
        chunks.extend(
            chunk_ticket(
                ticket,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
        )

    written = store.upsert(chunks)
    duration = time.perf_counter() - started

    logger.info(
        "ingest complete: %d tickets, %d chunks in %.2fs", len(tickets), written, duration
    )
    return IngestStats(
        files_read=files_read,
        tickets_indexed=len(tickets),
        chunks_written=written,
        duration_seconds=round(duration, 3),
        collection=store.collection_name,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Index support tickets into Chroma.")
    parser.add_argument("--reset", action="store_true", help="drop the collection first")
    parser.add_argument("--documents", type=Path, default=None, help="override documents dir")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    store = VectorStore(settings.chroma_path, settings.collection_name)

    stats = ingest(store, documents_dir=args.documents, reset=args.reset)
    print(
        f"\n  {stats.tickets_indexed} tickets -> {stats.chunks_written} chunks "
        f"in {stats.duration_seconds}s"
        f"\n  collection '{stats.collection}' now holds {store.count()} chunks"
        f"\n  path: {settings.chroma_path}\n"
    )


if __name__ == "__main__":
    main()
