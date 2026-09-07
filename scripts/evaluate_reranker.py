#!/usr/bin/env python3
"""Cross-Encoder Reranker Evaluation & Demonstration Script.

This script demonstrates and verifies how the 2-Stage Retrieval Pipeline works:
  - Stage 1: Candidate Generation (Dense vector similarity + BM25 Lexical + Hybrid RRF)
  - Stage 2: Cross-Encoder Reranker (Joint query-document cross-attention scoring)

It prints:
  1. Detailed comparison table across all 12 Golden Set support questions.
  2. Rank movements: how Cross-Encoder promotes relevant chunks to Rank 1.
  3. p50 Latency comparison (Dense vs Hybrid vs Cross-Encoder).
  4. Interactive test mode to test any question in real time.

Usage:
  python scripts/evaluate_reranker.py
  python scripts/evaluate_reranker.py --interactive
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

# Setup project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import VectorStore


def load_golden_set(path: Path) -> list[dict]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def run_benchmark():
    settings = get_settings()
    golden_path = PROJECT_ROOT / "golden_set.jsonl"
    golden_set = load_golden_set(golden_path)

    store = VectorStore(settings.chroma_path, "support_articles_table_aware")
    retriever = Retriever(store, top_k=3, min_relevance=0.0)
    reranker = retriever._get_reranker()

    print("=" * 95)
    print("  STAGE 2 CROSS-ENCODER RERANKER BENCHMARK & DIAGNOSTICS")
    print("=" * 95)
    print(f"  Golden Questions : {len(golden_set)} items from {golden_path.name}")
    print(f"  Vector Store     : {store.collection_name} ({store.count()} chunks)")
    print(f"  Reranker Backend : {reranker.backend} ({reranker.model_name})")
    print("-" * 95)

    dense_hits = 0
    hybrid_hits = 0
    rerank_hits = 0
    dense_lats = []
    hybrid_lats = []
    rerank_lats = []

    print(f"{'ID':<4} | {'Target':<12} | {'Dense (Stg 1)':<15} | {'Hybrid RRF':<15} | {'Cross-Encoder (Stg 2)':<22} | {'Status'}")
    print("-" * 95)

    for item in golden_set:
        qid = item["id"]
        q = item["question"]
        target = item["correct_chunk_id"]

        # Dense
        t0 = time.perf_counter()
        d_chunks = retriever.retrieve(q, top_k=3, mode="dense")
        d_lat = (time.perf_counter() - t0) * 1000
        dense_lats.append(d_lat)
        d_cids = [c.chunk_id for c in d_chunks]
        d_hit = target in d_cids
        d_rank = d_cids.index(target) + 1 if d_hit else None
        if d_hit:
            dense_hits += 1

        # Hybrid RRF
        t0 = time.perf_counter()
        h_chunks = retriever.retrieve(q, top_k=3, mode="hybrid")
        h_lat = (time.perf_counter() - t0) * 1000
        hybrid_lats.append(h_lat)
        h_cids = [c.chunk_id for c in h_chunks]
        h_hit = target in h_cids
        h_rank = h_cids.index(target) + 1 if h_hit else None
        if h_hit:
            hybrid_hits += 1

        # Cross-Encoder Rerank
        t0 = time.perf_counter()
        r_chunks = retriever.retrieve(q, top_k=3, mode="rerank")
        r_lat = (time.perf_counter() - t0) * 1000
        rerank_lats.append(r_lat)
        r_cids = [c.chunk_id for c in r_chunks]
        r_hit = target in r_cids
        r_rank = r_cids.index(target) + 1 if r_hit else None
        if r_hit:
            rerank_hits += 1

        d_str = f"Rank {d_rank}" if d_hit else "MISS"
        h_str = f"Rank {h_rank}" if h_hit else "MISS"
        r_top_score = r_chunks[0].cross_encoder_score if (r_chunks and r_chunks[0].cross_encoder_score) else 0.0
        r_str = f"Rank {r_rank} (sc: {r_top_score:.3f})" if r_hit else "MISS"

        if not d_hit and r_hit:
            status = "FIXED BY RERANKER"
        elif d_hit and r_hit and r_rank == 1 and d_rank != 1:
            status = "PROMOTED TO #1"
        elif r_hit:
            status = "PASSED"
        else:
            status = "BROKEN"

        print(f"{qid:<4} | {target:<12} | {d_str:<15} | {h_str:<15} | {r_str:<22} | {status}")

    print("-" * 95)
    total = len(golden_set)
    print("\nSUMMARY METRICS:")
    print(f"  1. Dense Baseline Hit-Rate@3      : {dense_hits}/{total} ({dense_hits/total*100:.1f}%) | p50 Latency: {statistics.median(dense_lats):.2f} ms")
    print(f"  2. Hybrid BM25+RRF Hit-Rate@3     : {hybrid_hits}/{total} ({hybrid_hits/total*100:.1f}%) | p50 Latency: {statistics.median(hybrid_lats):.2f} ms")
    print(f"  3. Cross-Encoder Rerank Hit-Rate@3: {rerank_hits}/{total} ({rerank_hits/total*100:.1f}%) | p50 Latency: {statistics.median(rerank_lats):.2f} ms")
    print("=" * 95)


def run_interactive():
    settings = get_settings()
    store = VectorStore(settings.chroma_path, "support_articles_table_aware")
    retriever = Retriever(store, top_k=3, min_relevance=0.0)

    print("\n" + "=" * 80)
    print("  INTERACTIVE CROSS-ENCODER RERANKER INSPECTOR")
    print("  Type any support query or error code to see side-by-side rankings.")
    print("  Type 'exit' or 'quit' to stop.")
    print("=" * 80)

    while True:
        try:
            query = input("\nQuery > ").strip()
            if not query or query.lower() in {"exit", "quit", "q"}:
                break

            print(f"\n--- Candidate Generation & Cross-Encoder Reranking for: '{query}' ---")
            
            # Fetch dense, bm25, hybrid, and rerank
            dense_chunks = retriever.retrieve(query, top_k=3, mode="dense")
            hybrid_chunks = retriever.retrieve(query, top_k=3, mode="hybrid")
            rerank_chunks = retriever.retrieve(query, top_k=3, mode="rerank")

            print("\n[Stage 1: Dense Cosine Similarity Top-3]")
            for i, c in enumerate(dense_chunks, 1):
                print(f"  #{i} {c.chunk_id:<12} Score: {c.dense_score or c.score:.4f} | {c.text[:80]}...")

            print("\n[Stage 1: Hybrid RRF Top-3]")
            for i, c in enumerate(hybrid_chunks, 1):
                print(f"  #{i} {c.chunk_id:<12} RRF: {c.rrf_score:.5f} (Dense: #{c.dense_rank}, BM25: #{c.bm25_rank})")

            print("\n[Stage 2: Cross-Encoder Neural Rerank Top-3]")
            for i, c in enumerate(rerank_chunks, 1):
                print(f"  #{i} {c.chunk_id:<12} Cross-Encoder Score: {c.cross_encoder_score:.4f} (Dense: #{c.dense_rank}, BM25: #{c.bm25_rank})")
                print(f"      Snippet: {c.text[:100]}...")

        except (KeyboardInterrupt, EOFError):
            break
    print("\nGoodbye!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Cross-Encoder Reranker")
    parser.add_argument("--interactive", "-i", action="store_true", help="Launch interactive query inspector")
    args = parser.parse_args()

    if args.interactive:
        run_interactive()
    else:
        run_benchmark()
