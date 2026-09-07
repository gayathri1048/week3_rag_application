#!/usr/bin/env python3
"""Week-4 Practical Evaluation Script: Failure Separation, Baseline vs Hybrid BM25+RRF (k=60),
p50 Latency Benchmark, and MMR Diversity Evaluation.

Requirements:
  1. Golden set: 12 real support questions with known-correct chunk_id.
  2. Baseline hit-rate@3 (dense retrieval).
  3. Failure labeling: R (retrieval fetched bad context), G (generation failed on good context), Not-In-Corpus with 1 line of evidence per failure.
  4. Exactly ONE retrieval change: BM25 + RRF fusion (k=60).
  5. Re-measure hit-rate@3 on the SAME 12 questions and report before -> after, plus p50 latency per query before -> after.
  6. Per-question fixed / unfixed / still-broken table.
  7. Bonus challenge: MMR diversity vs hit-rate trade-off analysis.

Usage:
  python scripts/evaluate_retrieval_week4.py
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

# Set up project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from app.config import get_settings
from app.retrieval.bm25 import tokenize
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


def measure_p50_latency(
    retriever: Retriever,
    query: str,
    mode: str,
    top_k: int = 3,
    iterations: int = 10,
    mmr_lambda: float = 0.7,
) -> float:
    """Measure median (p50) query latency in milliseconds over multiple runs."""
    # Warm up
    retriever.retrieve(query, top_k=top_k, mode=mode, mmr_lambda=mmr_lambda)

    durations: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        retriever.retrieve(query, top_k=top_k, mode=mode, mmr_lambda=mmr_lambda)
        t1 = time.perf_counter()
        durations.append((t1 - t0) * 1000.0)

    return statistics.median(durations)


def run_evaluation():
    settings = get_settings()
    golden_path = PROJECT_ROOT / "golden_set.jsonl"
    golden_set = load_golden_set(golden_path)

    # We evaluate on support_articles_table_aware collection
    store = VectorStore(settings.chroma_path, "support_articles_table_aware")
    retriever = Retriever(store, top_k=3, min_relevance=0.0)

    print("=" * 85, flush=True)
    print("WEEK 4 PRACTICAL — RETRIEVAL DEBUGGING & FAILURE SEPARATION BENCHMARK", flush=True)
    print("=" * 85, flush=True)
    print(f"Loaded {len(golden_set)} golden questions from {golden_path.name}", flush=True)
    print(f"Active collection: {store.collection_name} ({store.count()} chunks)", flush=True)
    print("-" * 85, flush=True)

    # -------------------------------------------------------------------------
    # 1. BASELINE EVALUATION (Dense Vector Retrieval)
    # -------------------------------------------------------------------------
    print("\n>>> Phase 1: Evaluating BASELINE (Dense Cosine Similarity, top_k=3)...", flush=True)
    baseline_results = []
    baseline_latencies = []
    baseline_hits = 0

    for item in golden_set:
        qid = item["id"]
        q = item["question"]
        target = item["correct_chunk_id"]

        lat = measure_p50_latency(retriever, q, mode="dense", top_k=3, iterations=10)
        baseline_latencies.append(lat)

        chunks = retriever.retrieve(q, top_k=3, mode="dense")
        retrieved_ids = [c.chunk_id for c in chunks]
        is_hit = target in retrieved_ids
        rank = retrieved_ids.index(target) + 1 if is_hit else None
        if is_hit:
            baseline_hits += 1

        baseline_results.append({
            "id": qid,
            "question": q,
            "correct_chunk_id": target,
            "has_exact_token": item.get("has_exact_token", False),
            "hit": is_hit,
            "rank": rank,
            "retrieved_chunk_ids": retrieved_ids,
            "scores": [c.score for c in chunks],
            "p50_ms": round(lat, 2),
        })
        print(f"  {qid}: {'HIT' if is_hit else 'MISS'} (Rank: {rank or '-'}) | p50: {lat:.2f}ms | target: {target} | got: {retrieved_ids}", flush=True)

    baseline_hit_rate = baseline_hits / len(golden_set)
    baseline_p50_overall = statistics.median(baseline_latencies)

    print(f"\nBaseline Hit-Rate@3: {baseline_hits}/{len(golden_set)} ({baseline_hit_rate * 100:.1f}%)", flush=True)
    print(f"Baseline Overall p50 Latency: {baseline_p50_overall:.2f} ms", flush=True)

    # -------------------------------------------------------------------------
    # 2. FAILURE SEPARATION (Inspection View Analysis)
    # -------------------------------------------------------------------------
    print("\n>>> Phase 2: Inspection View Failure Labeling (R / G / Not-In-Corpus)...", flush=True)
    failures = []
    r_count, g_count, nic_count = 0, 0, 0

    # Check if target chunk exists in corpus
    all_chunks = store.get_all()
    corpus_chunk_ids = {c["chunk_id"]: c["text"] for c in all_chunks}

    for b in baseline_results:
        if not b["hit"]:
            qid = b["id"]
            target = b["correct_chunk_id"]
            retrieved = b["retrieved_chunk_ids"]

            if target not in corpus_chunk_ids:
                label = "Not-In-Corpus"
                evidence = f"Target chunk {target} does not exist in the indexed database."
                nic_count += 1
            else:
                # Target exists in corpus, but was not in top-3
                label = "R"
                evidence = (
                    f"Retrieval returned {retrieved} (top-1 {retrieved[0] if retrieved else 'None'}), "
                    f"missing target {target} which contains the required answer."
                )
                r_count += 1

            failures.append({
                "id": qid,
                "question": b["question"],
                "target_chunk_id": target,
                "label": label,
                "retrieved": retrieved,
                "evidence": evidence,
            })

    print(f"Failure Tally: R={r_count} | G={g_count} | Not-In-Corpus={nic_count} (Total Misses={len(failures)})", flush=True)
    for f_item in failures:
        print(f"  [{f_item['label']}] {f_item['id']}: {f_item['evidence']}", flush=True)

    # -------------------------------------------------------------------------
    # 3. AFTER RETRIEVAL CHANGE (BM25 + RRF Fusion k=60)
    # -------------------------------------------------------------------------
    print("\n>>> Phase 3: Evaluating ONE CHANGE: BM25 + RRF Fusion (k=60, top_k=3)...", flush=True)
    hybrid_results = []
    hybrid_latencies = []
    hybrid_hits = 0

    for item in golden_set:
        qid = item["id"]
        q = item["question"]
        target = item["correct_chunk_id"]

        lat = measure_p50_latency(retriever, q, mode="hybrid", top_k=3, iterations=10)
        hybrid_latencies.append(lat)

        chunks = retriever.retrieve(q, top_k=3, mode="hybrid")
        retrieved_ids = [c.chunk_id for c in chunks]
        is_hit = target in retrieved_ids
        rank = retrieved_ids.index(target) + 1 if is_hit else None
        if is_hit:
            hybrid_hits += 1

        hybrid_results.append({
            "id": qid,
            "question": q,
            "correct_chunk_id": target,
            "has_exact_token": item.get("has_exact_token", False),
            "hit": is_hit,
            "rank": rank,
            "retrieved_chunk_ids": retrieved_ids,
            "dense_ranks": [c.dense_rank for c in chunks],
            "bm25_ranks": [c.bm25_rank for c in chunks],
            "rrf_scores": [c.rrf_score for c in chunks],
            "p50_ms": round(lat, 2),
        })
        print(f"  {qid}: {'HIT' if is_hit else 'MISS'} (Rank: {rank or '-'}) | p50: {lat:.2f}ms | target: {target} | got: {retrieved_ids}", flush=True)

    hybrid_hit_rate = hybrid_hits / len(golden_set)
    hybrid_p50_overall = statistics.median(hybrid_latencies)

    print(f"\nHybrid Hit-Rate@3: {hybrid_hits}/{len(golden_set)} ({hybrid_hit_rate * 100:.1f}%)", flush=True)
    print(f"Hybrid Overall p50 Latency: {hybrid_p50_overall:.2f} ms", flush=True)

    # -------------------------------------------------------------------------
    # 4. PER-QUESTION FIXED / UNFIXED / STILL-BROKEN TABLE
    # -------------------------------------------------------------------------
    print("\n>>> Phase 4: Per-Question Comparison Table...", flush=True)
    comparison_table = []
    fixed_count, unfixed_count, passed_count = 0, 0, 0

    for b, h in zip(baseline_results, hybrid_results):
        qid = b["id"]
        if not b["hit"] and h["hit"]:
            status = "FIXED"
            fixed_count += 1
        elif not b["hit"] and not h["hit"]:
            status = "STILL_BROKEN"
            unfixed_count += 1
        elif b["hit"] and h["hit"]:
            status = "PASSED_BOTH"
            passed_count += 1
        else:
            status = "REGRESSED"

        comparison_table.append({
            "id": qid,
            "question": b["question"],
            "target": b["correct_chunk_id"],
            "exact_token": b["has_exact_token"],
            "baseline_hit": b["hit"],
            "baseline_rank": b["rank"] if b["hit"] else "-",
            "baseline_p50": b["p50_ms"],
            "hybrid_hit": h["hit"],
            "hybrid_rank": h["rank"] if h["hit"] else "-",
            "hybrid_p50": h["p50_ms"],
            "status": status,
        })

    print(f"{'Q#':<4} | {'Exact Token':<11} | {'Target':<11} | {'Baseline (Hit/Rank/ms)':<22} | {'Hybrid (Hit/Rank/ms)':<22} | {'Outcome'}", flush=True)
    print("-" * 88, flush=True)
    for c in comparison_table:
        b_str = f"{'HIT' if c['baseline_hit'] else 'MISS':<4} (R:{str(c['baseline_rank']):<2}, {c['baseline_p50']}ms)"
        h_str = f"{'HIT' if c['hybrid_hit'] else 'MISS':<4} (R:{str(c['hybrid_rank']):<2}, {c['hybrid_p50']}ms)"
        token_str = "YES" if c["exact_token"] else "NO"
        print(f"{c['id']:<4} | {token_str:<11} | {c['target']:<11} | {b_str:<22} | {h_str:<22} | {c['status']}", flush=True)

    # -------------------------------------------------------------------------
    # 5. BONUS CHALLENGE: MMR Evaluation
    # -------------------------------------------------------------------------
    print("\n>>> Phase 5: Bonus Challenge — MMR Diversity vs Hit-Rate Analysis...", flush=True)
    err_query = "I am getting error code ERR-4032 during subscription migration, what is the fix?"
    lambdas = [1.0, 0.7, 0.5, 0.3]
    mmr_records = []

    def compute_diversity(chunks: list) -> float:
        """Compute pairwise Jaccard distance between retrieved chunks (higher = more diverse)."""
        if len(chunks) < 2:
            return 1.0
        token_sets = [set(tokenize(c.text)) for c in chunks]
        distances = []
        for i in range(len(token_sets)):
            for j in range(i + 1, len(token_sets)):
                s1, s2 = token_sets[i], token_sets[j]
                inter = len(s1 & s2)
                union = len(s1 | s2)
                jaccard = inter / union if union > 0 else 0.0
                distances.append(1.0 - jaccard)
        return statistics.mean(distances) if distances else 0.0

    print(f"Query: '{err_query}' (Target: BM-002::t3)", flush=True)
    for lam in lambdas:
        mmr_chunks = retriever.retrieve(err_query, top_k=3, mode="mmr", mmr_lambda=lam)
        cids = [c.chunk_id for c in mmr_chunks]
        div = compute_diversity(mmr_chunks)
        hit = "BM-002::t3" in cids
        print(f"  lambda={lam:.1f}: Top-3={cids} | Target in Top-3: {hit} | Diversity (avg Jaccard dist): {div:.4f}", flush=True)
        mmr_records.append({
            "lambda": lam,
            "top_3": cids,
            "target_hit": hit,
            "diversity": round(div, 4),
        })

    # Summary report dict
    report = {
        "baseline": {
            "hit_rate_at_3": f"{baseline_hits}/{len(golden_set)} ({baseline_hit_rate * 100:.1f}%)",
            "p50_latency_ms": round(baseline_p50_overall, 2),
            "failures": failures,
            "results": baseline_results,
        },
        "hybrid": {
            "hit_rate_at_3": f"{hybrid_hits}/{len(golden_set)} ({hybrid_hit_rate * 100:.1f}%)",
            "p50_latency_ms": round(hybrid_p50_overall, 2),
            "results": hybrid_results,
        },
        "delta": {
            "hit_rate_before": f"{baseline_hits}/{len(golden_set)}",
            "hit_rate_after": f"{hybrid_hits}/{len(golden_set)}",
            "p50_before_ms": round(baseline_p50_overall, 2),
            "p50_after_ms": round(hybrid_p50_overall, 2),
            "latency_overhead_ms": round(hybrid_p50_overall - baseline_p50_overall, 2),
            "fixed_count": fixed_count,
            "unfixed_count": unfixed_count,
        },
        "comparison_table": comparison_table,
        "mmr_bonus": mmr_records,
    }

    out_file = PROJECT_ROOT / "scripts" / "week4_eval_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved full benchmark data to: {out_file}", flush=True)
    return report


if __name__ == "__main__":
    run_evaluation()
