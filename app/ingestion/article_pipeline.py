"""Ingest orchestration for Week-3 help-centre articles.

Usage
-----
Index 6 new articles ONLY (Strategy 1 — paragraph chunker):
    python -m app.ingestion.article_pipeline --strategy paragraph

Index 6 new articles ONLY (Strategy 2 — table-aware chunker):
    python -m app.ingestion.article_pipeline --strategy table_aware

Both strategies write to separate Chroma collections so they can be compared
on the same 8 evaluation questions without interference.

Collections
  support_articles_paragraph   ← Strategy 1
  support_articles_table_aware ← Strategy 2

The full historical corpus is NOT re-indexed here (per Week-3 requirement 6).
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from ..config import get_settings
from ..retrieval.vector_store import VectorStore
from ..schemas import IngestStats
from .article_loader import load_articles
from .chunker import chunk_article
from .table_chunker import chunk_article_table_aware

logger = logging.getLogger(__name__)

STRATEGY_COLLECTION = {
    "paragraph": "support_articles_paragraph",
    "table_aware": "support_articles_table_aware",
}


def ingest_articles(
    store: VectorStore,
    articles_dir: Path | None = None,
    strategy: str = "paragraph",
    reset: bool = False,
) -> IngestStats:
    """Read every article file, chunk it with the chosen strategy, upsert into the store.

    NOTE: only the 6 new help-centre articles are indexed here.
    The historical ticket corpus is untouched (Week-3 requirement 6).
    """
    settings = get_settings()
    source_dir = articles_dir or (settings.documents_dir / "help_articles")
    started = time.perf_counter()

    if reset:
        store.reset()

    articles, files_read = load_articles(source_dir)

    chunk_fn = chunk_article if strategy == "paragraph" else chunk_article_table_aware

    from .chunker import Chunk  # shared Chunk dataclass shape
    chunks: list[Chunk] = []
    for article in articles:
        chunks.extend(
            chunk_fn(
                article,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
        )

    written = store.upsert(chunks)
    duration = time.perf_counter() - started

    logger.info(
        "article ingest complete: %d articles, %d chunks in %.2fs (strategy=%s)",
        len(articles), written, duration, strategy,
    )
    return IngestStats(
        files_read=files_read,
        tickets_indexed=len(articles),
        chunks_written=written,
        duration_seconds=round(duration, 3),
        collection=store.collection_name,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Index Week-3 help-centre articles into Chroma.")
    parser.add_argument(
        "--strategy",
        choices=["paragraph", "table_aware"],
        default="paragraph",
        help="Chunking strategy to use.",
    )
    parser.add_argument("--reset", action="store_true", help="drop the collection first")
    parser.add_argument("--articles", type=Path, default=None, help="override articles dir")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    settings = get_settings()
    collection_name = STRATEGY_COLLECTION[args.strategy]
    store = VectorStore(settings.chroma_path, collection_name)

    stats = ingest_articles(store, articles_dir=args.articles, strategy=args.strategy, reset=args.reset)
    print(
        f"\n  Strategy    : {args.strategy}"
        f"\n  Articles    : {stats.tickets_indexed}"
        f"\n  Chunks      : {stats.chunks_written}"
        f"\n  Duration    : {stats.duration_seconds}s"
        f"\n  Collection  : '{stats.collection}'"
        f"\n  Path        : {settings.chroma_path}\n"
    )


if __name__ == "__main__":
    main()
