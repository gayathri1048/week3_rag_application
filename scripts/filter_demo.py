#!/usr/bin/env python3
"""Week-3 metadata filter demo script.

Shows how filtering on product_area changes the top-1 result.
Pastes both unfiltered and filtered result lists with scores.

Usage:
    python scripts/filter_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings
from app.retrieval.vector_store import VectorStore
from app.retrieval.retriever import Retriever
from app.schemas import ArticleFilters

# Use the table_aware collection for the demo (can be changed to "paragraph")
COLLECTION = "support_articles_table_aware"
QUERY = "What does error code ERR-4032 mean and what is the fix?"
TOP_K = 5


def print_result_list(label: str, chunks: list) -> None:
    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}")
    print(f"{'Rank':<6} {'chunk_id':<22} {'article_id':<12} {'product_area':<14} {'score'}")
    print("-"*65)
    for i, c in enumerate(chunks, 1):
        print(
            f"{i:<6} {c.chunk_id:<22} {(c.article_id or '—'):<12} "
            f"{(c.product_area or '—'):<14} {c.score:.4f}"
        )
    if chunks:
        print(f"\nTop-1 chunk_id : {chunks[0].chunk_id}")
        print(f"Top-1 article  : {chunks[0].article_id}")
        print(f"Top-1 score    : {chunks[0].score:.4f}")


def main() -> None:
    settings = get_settings()
    store = VectorStore(settings.chroma_path, COLLECTION)
    retriever = Retriever(store, top_k=TOP_K, min_relevance=0.0)

    print(f"\nQuery: {QUERY!r}")
    print(f"Collection: {COLLECTION}")

    # --- Unfiltered ---
    unfiltered = retriever.retrieve(QUERY, top_k=TOP_K)
    print_result_list("UNFILTERED (no product_area filter)", unfiltered)

    # --- Filtered: product_area = "Account" ---
    filtered = retriever.retrieve(
        QUERY,
        top_k=TOP_K,
        article_filters=ArticleFilters(product_area="Account"),
    )
    print_result_list("FILTERED: product_area = 'Account'", filtered)

    print("\n--- Interpretation ---")
    if unfiltered and filtered:
        if unfiltered[0].chunk_id != filtered[0].chunk_id:
            print(f"Top-1 changed: {unfiltered[0].chunk_id!r} → {filtered[0].chunk_id!r}")
        else:
            print("Top-1 did not change — try a different query or filter value.")


if __name__ == "__main__":
    main()
