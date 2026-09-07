"""Inspection and evaluation routes for Week 4 retrieval debugging workbench."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Depends, Query
from pydantic import BaseModel, Field

from ...config import get_settings
from ...dependencies import get_retriever, get_vector_store
from ...retrieval.bm25 import tokenize
from ...retrieval.retriever import Retriever
from ...retrieval.vector_store import VectorStore
from ...schemas import RetrievedChunk

router = APIRouter(prefix="/inspection", tags=["inspection"])


class GoldenItem(BaseModel):
    id: str
    question: str
    correct_chunk_id: str
    has_exact_token: bool
    category: str
    article_id: str


class CompareRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=3, ge=1, le=25)
    mmr_lambda: float = Field(default=0.7, ge=0.0, le=1.0)


class CompareResult(BaseModel):
    query: str
    dense_chunks: list[RetrievedChunk]
    bm25_chunks: list[RetrievedChunk]
    hybrid_chunks: list[RetrievedChunk]
    mmr_chunks: list[RetrievedChunk]
    rerank_chunks: list[RetrievedChunk]


class DirectRerankRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    passages: list[str] = Field(min_length=1)
    top_k: int = Field(default=3, ge=1)


class DirectRerankHit(BaseModel):
    rank: int
    score: float
    text: str


class DirectRerankResponse(BaseModel):
    query: str
    backend: str
    results: list[DirectRerankHit]


class EvaluateResponse(BaseModel):
    baseline_hit_rate: str
    baseline_hits: int
    baseline_total: int
    baseline_p50_ms: float
    hybrid_hit_rate: str
    hybrid_hits: int
    hybrid_total: int
    hybrid_p50_ms: float
    rerank_hit_rate: str
    rerank_hits: int
    rerank_total: int
    rerank_p50_ms: float
    fixed_count: int
    unfixed_count: int
    failures: list[dict[str, Any]]
    details: list[dict[str, Any]]


def _load_golden_set() -> list[dict[str, Any]]:
    path = get_settings().chroma_path.parent / "golden_set.jsonl"
    if not path.exists():
        path = Path(__file__).resolve().parent.parent.parent.parent / "golden_set.jsonl"
    items = []
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    items.append(json.loads(line))
    return items


@router.get("/golden-set", response_model=list[GoldenItem])
def get_golden_set() -> list[GoldenItem]:
    """Return all 12 questions in the golden evaluation set."""
    raw = _load_golden_set()
    return [GoldenItem(**item) for item in raw]


@router.get("/corpus")
def get_corpus(retriever: Retriever = Depends(get_retriever)) -> dict[str, Any]:
    """Return all indexed documents and their chunk breakdown from the vector store."""
    all_chunks = retriever.store.get_all()
    grouped: dict[str, dict[str, Any]] = {}
    for c in all_chunks:
        meta = c.get("metadata", {})
        doc_id = str(meta.get("article_id") or meta.get("ticket_id") or c["chunk_id"].split("::")[0])
        if doc_id not in grouped:
            grouped[doc_id] = {
                "id": doc_id,
                "title": meta.get("subject", "(untitled)"),
                "source_file": meta.get("source_file", f"{doc_id}.md"),
                "product_area": meta.get("product_area", meta.get("category", "General")),
                "last_updated": meta.get("last_updated", "N/A"),
                "tags": meta.get("tags", ""),
                "chunks": [],
            }
        grouped[doc_id]["chunks"].append({
            "chunk_id": c["chunk_id"],
            "chunk_index": meta.get("chunk_index", 0),
            "text": c.get("text", ""),
            "preview": c.get("text", "")[:180] + "..." if len(c.get("text", "")) > 180 else c.get("text", ""),
        })

    docs = list(grouped.values())
    for d in docs:
        d["chunks"].sort(key=lambda x: int(x.get("chunk_index", 0)))
        d["chunk_count"] = len(d["chunks"])

    docs.sort(key=lambda x: x["id"])

    uploaded_images = []
    settings = get_settings()
    img_dir = settings.documents_dir / "uploaded_images"
    if img_dir.exists():
        for p in sorted(img_dir.glob("*"), key=lambda f: f.stat().st_mtime, reverse=True):
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
                uploaded_images.append({
                    "filename": p.name,
                    "size_kb": round(p.stat().st_size / 1024, 1),
                    "path": str(p),
                })

    return {
        "collection": retriever.store.collection_name,
        "total_documents": len(docs),
        "total_chunks": len(all_chunks),
        "documents": docs,
        "uploaded_images": uploaded_images,
    }


@router.post("/compare", response_model=CompareResult)
def compare_retrieval(
    req: CompareRequest,
    retriever: Retriever = Depends(get_retriever),
) -> CompareResult:
    """Run side-by-side retrieval across Dense, BM25, Hybrid RRF, MMR, and Cross-Encoder Reranker."""
    dense_hits = retriever.retrieve(req.query, top_k=req.top_k, mode="dense")
    bm25_hits = retriever.retrieve(req.query, top_k=req.top_k, mode="bm25")
    hybrid_hits = retriever.retrieve(req.query, top_k=req.top_k, mode="hybrid")
    mmr_hits = retriever.retrieve(req.query, top_k=req.top_k, mode="mmr", mmr_lambda=req.mmr_lambda)
    rerank_hits = retriever.retrieve(req.query, top_k=req.top_k, mode="rerank")

    return CompareResult(
        query=req.query,
        dense_chunks=dense_hits,
        bm25_chunks=bm25_hits,
        hybrid_chunks=hybrid_hits,
        mmr_chunks=mmr_hits,
        rerank_chunks=rerank_hits,
    )


@router.post("/rerank", response_model=DirectRerankResponse)
def direct_rerank(
    req: DirectRerankRequest,
    retriever: Retriever = Depends(get_retriever),
) -> DirectRerankResponse:
    """Directly rerank a custom list of text passages with the Cross-Encoder."""
    reranker = retriever._get_reranker()
    scores = reranker.compute_scores(req.query, req.passages)
    scored = [
        {"text": txt, "score": score, "original_index": i}
        for i, (txt, score) in enumerate(zip(req.passages, scores))
    ]
    scored.sort(key=lambda x: x["score"], reverse=True)
    results = [
        DirectRerankHit(rank=r, score=item["score"], text=item["text"])
        for r, item in enumerate(scored[: req.top_k], start=1)
    ]
    return DirectRerankResponse(query=req.query, backend=reranker.backend, results=results)


@router.post("/evaluate", response_model=EvaluateResponse)
def evaluate_retrieval(
    retriever: Retriever = Depends(get_retriever),
) -> EvaluateResponse:
    """Run full evaluation comparing Baseline Dense vs BM25+RRF Hybrid vs Cross-Encoder on the Golden Set."""
    golden_set = _load_golden_set()
    all_chunks = retriever.store.get_all()
    corpus_ids = {c["chunk_id"] for c in all_chunks}

    baseline_hits = 0
    hybrid_hits = 0
    rerank_hits = 0
    baseline_latencies = []
    hybrid_latencies = []
    rerank_latencies = []
    failures = []
    details = []

    for item in golden_set:
        qid = item["id"]
        q = item["question"]
        target = item["correct_chunk_id"]

        # 1. Dense retrieval timing & accuracy
        t0 = time.perf_counter()
        d_chunks = retriever.retrieve(q, top_k=3, mode="dense")
        t1 = time.perf_counter()
        d_lat = (t1 - t0) * 1000.0
        baseline_latencies.append(d_lat)

        d_cids = [c.chunk_id for c in d_chunks]
        d_hit = target in d_cids
        d_rank = d_cids.index(target) + 1 if d_hit else None
        if d_hit:
            baseline_hits += 1

        # 2. Hybrid retrieval timing & accuracy
        t0 = time.perf_counter()
        h_chunks = retriever.retrieve(q, top_k=3, mode="hybrid")
        t1 = time.perf_counter()
        h_lat = (t1 - t0) * 1000.0
        hybrid_latencies.append(h_lat)

        h_cids = [c.chunk_id for c in h_chunks]
        h_hit = target in h_cids
        h_rank = h_cids.index(target) + 1 if h_hit else None
        if h_hit:
            hybrid_hits += 1

        # 3. Cross-Encoder Rerank timing & accuracy
        t0 = time.perf_counter()
        r_chunks = retriever.retrieve(q, top_k=3, mode="rerank")
        t1 = time.perf_counter()
        r_lat = (t1 - t0) * 1000.0
        rerank_latencies.append(r_lat)

        r_cids = [c.chunk_id for c in r_chunks]
        r_hit = target in r_cids
        r_rank = r_cids.index(target) + 1 if r_hit else None
        if r_hit:
            rerank_hits += 1

        # Failure labeling for baseline misses
        if not d_hit:
            if target not in corpus_ids:
                label = "Not-In-Corpus"
                evidence = f"Target chunk {target} is not present in indexed collection."
            else:
                label = "R"
                evidence = (
                    f"Retrieval returned {d_cids} (top-1 {d_cids[0] if d_cids else 'None'}), "
                    f"missing target {target} in top-3."
                )
            failures.append({
                "id": qid,
                "question": q,
                "target_chunk_id": target,
                "label": label,
                "evidence": evidence,
                "retrieved": d_cids,
            })

        status = "PASSED_BOTH" if (d_hit and h_hit) else ("FIXED" if (not d_hit and h_hit) else ("STILL_BROKEN" if (not d_hit and not h_hit) else "REGRESSED"))

        details.append({
            "id": qid,
            "question": q,
            "target": target,
            "has_exact_token": item.get("has_exact_token", False),
            "category": item.get("category", "general"),
            "baseline_hit": d_hit,
            "baseline_rank": d_rank,
            "baseline_p50_ms": round(d_lat, 2),
            "baseline_chunks": d_cids,
            "hybrid_hit": h_hit,
            "hybrid_rank": h_rank,
            "hybrid_p50_ms": round(h_lat, 2),
            "hybrid_chunks": h_cids,
            "rerank_hit": r_hit,
            "rerank_rank": r_rank,
            "rerank_p50_ms": round(r_lat, 2),
            "rerank_chunks": r_cids,
            "status": status,
        })

    fixed_cnt = sum(1 for d in details if d["status"] == "FIXED")
    unfixed_cnt = sum(1 for d in details if d["status"] == "STILL_BROKEN")

    return EvaluateResponse(
        baseline_hit_rate=f"{baseline_hits}/{len(golden_set)} ({baseline_hits/len(golden_set)*100:.1f}%)",
        baseline_hits=baseline_hits,
        baseline_total=len(golden_set),
        baseline_p50_ms=round(statistics.median(baseline_latencies), 2),
        hybrid_hit_rate=f"{hybrid_hits}/{len(golden_set)} ({hybrid_hits/len(golden_set)*100:.1f}%)",
        hybrid_hits=hybrid_hits,
        hybrid_total=len(golden_set),
        hybrid_p50_ms=round(statistics.median(hybrid_latencies), 2),
        rerank_hit_rate=f"{rerank_hits}/{len(golden_set)} ({rerank_hits/len(golden_set)*100:.1f}%)",
        rerank_hits=rerank_hits,
        rerank_total=len(golden_set),
        rerank_p50_ms=round(statistics.median(rerank_latencies), 2),
        fixed_count=fixed_cnt,
        unfixed_count=unfixed_cnt,
        failures=failures,
        details=details,
    )


# =========================================================================
# WEEK 3 INGESTION & CHUNKING STRATEGY EVALUATION (Paragraph vs Table-Aware)
# =========================================================================

WEEK3_QUESTIONS = [
    {
        "q_num": 1,
        "question": "When does Phase 2 of the billing migration start and which accounts does it affect?",
        "correct_article": "BM-001",
        "key_phrase": "Phase 2",
        "type": "prose",
    },
    {
        "q_num": 2,
        "question": "How long do migration credits last before they expire?",
        "correct_article": "BM-003",
        "key_phrase": "12 months",
        "type": "prose",
    },
    {
        "q_num": 3,
        "question": "What HTTP header replaces X-Billing-Token in the Unified Billing Platform?",
        "correct_article": "BM-004",
        "key_phrase": "Authorization",
        "type": "prose",
    },
    {
        "q_num": 4,
        "question": "What does error code ERR-4032 mean and what is the fix?",
        "correct_article": "BM-002",
        "key_phrase": "ERR-4032",
        "type": "table_row",
    },
    {
        "q_num": 5,
        "question": "How should support handle ERR-4011 — a duplicate invoice detected during migration?",
        "correct_article": "BM-002",
        "key_phrase": "ERR-4011",
        "type": "table_row",
    },
    {
        "q_num": 6,
        "question": "A customer's saved card token could not be re-tokenised because the card expired. What error code appears and what must the customer do?",
        "correct_article": "BM-002",
        "key_phrase": "ERR-4001",
        "type": "table_row",
    },
    {
        "q_num": 7,
        "question": "What are the steps to fix ERR-4031 after SSO mapping is lost during migration?",
        "correct_article": "BM-005",
        "key_phrase": "ERR-4031",
        "type": "table_row",
    },
    {
        "q_num": 8,
        "question": "What are the sub-steps for resolving ERR-4032 on an enterprise account with a custom plan?",
        "correct_article": "BM-006",
        "key_phrase": "ERR-4032",
        "type": "table_row",
    },
]


@router.get("/week3/questions")
def get_week3_questions() -> list[dict[str, Any]]:
    """Return the 8 rubric questions for Week 3 chunking evaluation."""
    return WEEK3_QUESTIONS


@router.get("/week3/evaluation")
@router.post("/week3/evaluate")
def evaluate_week3_chunking() -> dict[str, Any]:
    """Evaluate Strategy 1 (Paragraph Chunker) vs Strategy 2 (Table-Aware Chunker) on 8 questions."""
    settings = get_settings()

    strategies = {
        "paragraph": "support_articles_paragraph",
        "table_aware": "support_articles_table_aware",
    }

    results_by_strategy = {}

    for strat_key, col_name in strategies.items():
        store = VectorStore(settings.chroma_path, col_name)
        retriever = Retriever(store, top_k=5, min_relevance=0.0, default_mode="dense")

        q_results = []
        hits = 0

        for item in WEEK3_QUESTIONS:
            t0 = time.perf_counter()
            chunks = retriever.retrieve(item["question"], top_k=5, mode="dense")
            elapsed = (time.perf_counter() - t0) * 1000

            retrieved_articles = [c.article_id or c.ticket_id for c in chunks]
            retrieved_chunk_ids = [c.chunk_id for c in chunks]
            scores = [round(c.score or 0.0, 4) for c in chunks]

            hit = item["correct_article"] in retrieved_articles
            if hit:
                hits += 1

            # rank of first hit
            rank = (retrieved_articles.index(item["correct_article"]) + 1) if hit else None

            q_results.append({
                "q_num": item["q_num"],
                "question": item["question"],
                "type": item["type"],
                "correct_article": item["correct_article"],
                "key_phrase": item["key_phrase"],
                "hit": hit,
                "rank": rank,
                "latency_ms": round(elapsed, 2),
                "retrieved_articles": retrieved_articles,
                "retrieved_chunk_ids": retrieved_chunk_ids,
                "scores": scores,
                "top1_chunk_id": retrieved_chunk_ids[0] if retrieved_chunk_ids else "None",
                "top1_score": scores[0] if scores else 0.0,
            })

        results_by_strategy[strat_key] = {
            "strategy": strat_key,
            "collection": col_name,
            "total": len(WEEK3_QUESTIONS),
            "hits": hits,
            "score": f"{hits}/{len(WEEK3_QUESTIONS)} ({hits/len(WEEK3_QUESTIONS)*100:.1f}%)",
            "results": q_results,
        }

    # Build side-by-side comparison table
    comparison = []
    for i, item in enumerate(WEEK3_QUESTIONS):
        p_res = results_by_strategy["paragraph"]["results"][i]
        t_res = results_by_strategy["table_aware"]["results"][i]

        score_delta = round(t_res["top1_score"] - p_res["top1_score"], 4)
        comparison.append({
            "q_num": item["q_num"],
            "question": item["question"],
            "type": item["type"],
            "correct_article": item["correct_article"],
            "paragraph_hit": p_res["hit"],
            "paragraph_rank": p_res["rank"],
            "paragraph_top1": p_res["top1_chunk_id"],
            "paragraph_score": p_res["top1_score"],
            "table_aware_hit": t_res["hit"],
            "table_aware_rank": t_res["rank"],
            "table_aware_top1": t_res["top1_chunk_id"],
            "table_aware_score": t_res["top1_score"],
            "score_delta": score_delta,
        })

    return {
        "paragraph": results_by_strategy["paragraph"],
        "table_aware": results_by_strategy["table_aware"],
        "comparison": comparison,
    }

