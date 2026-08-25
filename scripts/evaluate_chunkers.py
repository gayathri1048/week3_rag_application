#!/usr/bin/env python3
"""Week-3 evaluation script: run 8 known-answer questions search-only against
both chunking strategies and record hit-in-top-5 per question.

Usage
-----
# First, ingest with both strategies:
#   python -m app.ingestion.article_pipeline --strategy paragraph --reset
#   python -m app.ingestion.article_pipeline --strategy table_aware --reset

# Then run evaluation:
    python scripts/evaluate_chunkers.py

Output
------
Prints a per-question table and the final X/8 scores for both strategies.
Saves a machine-readable dump to scripts/eval_dump.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Make sure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings
from app.retrieval.vector_store import VectorStore
from app.retrieval.retriever import Retriever

# ================================================================ Questions ==
# Written from the articles BEFORE running search (per rubric warning).
# Each entry:  (question_text, correct_article_id, key_phrase_in_correct_chunk)

QUESTIONS: list[tuple[str, str, str]] = [
    # Q1 — prose (BM-001, migration timeline)
    (
        "When does Phase 2 of the billing migration start and which accounts does it affect?",
        "BM-001",
        "Phase 2",
    ),
    # Q2 — prose (BM-003, credit expiry)
    (
        "How long do migration credits last before they expire?",
        "BM-003",
        "12 months",
    ),
    # Q3 — prose (BM-004, auth header change)
    (
        "What HTTP header replaces X-Billing-Token in the Unified Billing Platform?",
        "BM-004",
        "Authorization",
    ),
    # Q4 — TABLE ROW (BM-002, ERR-4032)
    (
        "What does error code ERR-4032 mean and what is the fix?",
        "BM-002",
        "ERR-4032",
    ),
    # Q5 — TABLE ROW (BM-002, ERR-4011)
    (
        "How should support handle ERR-4011 — a duplicate invoice detected during migration?",
        "BM-002",
        "ERR-4011",
    ),
    # Q6 — TABLE ROW (BM-002, ERR-4001)
    (
        "A customer's saved card token could not be re-tokenised because the card expired. What error code appears and what must the customer do?",
        "BM-002",
        "ERR-4001",
    ),
    # Q7 — TABLE ROW (BM-005, ERR-4031 fix steps)
    (
        "What are the steps to fix ERR-4031 after SSO mapping is lost during migration?",
        "BM-005",
        "ERR-4031",
    ),
    # Q8 — TABLE ROW (BM-006, ERR-4032 sub-steps)
    (
        "What are the sub-steps for resolving ERR-4032 on an enterprise account with a custom plan?",
        "BM-006",
        "ERR-4032",
    ),
]

COLLECTION_MAP = {
    "paragraph":   "support_articles_paragraph",
    "table_aware": "support_articles_table_aware",
}

TOP_K = 5  # "hit-in-top-5" as specified by the task


# ================================================================ Evaluation =

def evaluate_strategy(strategy: str, questions: list[tuple[str, str, str]]) -> dict:
    """Run all questions against one strategy's collection. Return results dict."""
    settings = get_settings()
    collection_name = COLLECTION_MAP[strategy]
    store = VectorStore(settings.chroma_path, collection_name)
    retriever = Retriever(store, top_k=TOP_K, min_relevance=0.0)  # min=0 for eval

    results = []
    hits = 0

    for i, (question, correct_article, key_phrase) in enumerate(questions, start=1):
        chunks = retriever.retrieve(question, top_k=TOP_K)
        retrieved_ids = [c.article_id or c.ticket_id for c in chunks]
        retrieved_chunk_ids = [c.chunk_id for c in chunks]
        scores = [c.score for c in chunks]

        # Hit = correct article_id appears in top-5
        hit = correct_article in retrieved_ids
        if hit:
            hits += 1

        results.append({
            "q_num": i,
            "question": question,
            "correct_article": correct_article,
            "key_phrase": key_phrase,
            "hit": hit,
            "retrieved_article_ids": retrieved_ids,
            "retrieved_chunk_ids": retrieved_chunk_ids,
            "scores": [round(s, 4) for s in scores],
        })

    return {
        "strategy": strategy,
        "collection": collection_name,
        "total_questions": len(questions),
        "hits": hits,
        "score": f"{hits}/{len(questions)}",
        "results": results,
    }


def print_results(data: dict) -> None:
    print(f"\n{'='*70}")
    print(f"Strategy: {data['strategy']}  |  Collection: {data['collection']}")
    print(f"Score: {data['score']}")
    print(f"{'='*70}")
    print(f"{'Q#':<4} {'Hit':<5} {'Correct Article':<12} {'Top-1 Retrieved':<15} {'Score'}")
    print("-"*70)
    for r in data["results"]:
        top1 = r["retrieved_article_ids"][0] if r["retrieved_article_ids"] else "—"
        top1_score = r["scores"][0] if r["scores"] else 0.0
        hit_str = "✅" if r["hit"] else "❌"
        print(f"Q{r['q_num']:<3} {hit_str:<5} {r['correct_article']:<12} {top1:<15} {top1_score:.4f}")
    print()


def main() -> None:
    all_results = []

    for strategy in ("paragraph", "table_aware"):
        print(f"\nEvaluating strategy: {strategy} ...")
        try:
            data = evaluate_strategy(strategy, QUESTIONS)
            print_results(data)
            all_results.append(data)
        except Exception as exc:
            print(f"  ERROR: {exc}")
            print(f"  Have you run: python -m app.ingestion.article_pipeline --strategy {strategy} --reset")

    # Summary table
    if len(all_results) == 2:
        print("\n" + "="*70)
        print("SUMMARY — Hit-in-Top-5")
        print("="*70)
        print(f"{'Strategy':<15} {'Score':<10}")
        print("-"*30)
        for d in all_results:
            print(f"{d['strategy']:<15} {d['score']:<10}")
        print()

    # Save dump
    dump_path = PROJECT_ROOT / "scripts" / "eval_dump.json"
    dump_path.parent.mkdir(exist_ok=True)
    with open(dump_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Full dump saved to: {dump_path}")


if __name__ == "__main__":
    main()
