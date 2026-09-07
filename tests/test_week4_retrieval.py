"""Unit and integration tests for Week 4 retrieval enhancements."""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.retrieval.bm25 import BM25Index, compute_mmr_rerank, compute_rrf_fusion, tokenize


def test_tokenize_preserves_error_codes():
    tokens = tokenize("Customer hit ERR-4032 with header X-Billing-Token and code CUSTOM-01")
    assert "err-4032" in tokens
    assert "x-billing-token" in tokens
    assert "custom-01" in tokens


def test_bm25_exact_match_ranking():
    docs = [
        {"chunk_id": "c1", "text": "General billing help and invoice overviews."},
        {"chunk_id": "c2", "text": "Error ERR-4032 indicates subscription missing from UBP catalog."},
        {"chunk_id": "c3", "text": "SSO login issues and role mapping details."},
    ]
    index = BM25Index(docs)
    results = index.search("ERR-4032", top_k=2)
    assert len(results) >= 1
    assert results[0]["chunk_id"] == "c2"


def test_rrf_fusion_logic():
    dense_hits = [
        {"chunk_id": "doc1", "text": "Doc 1", "score": 0.8},
        {"chunk_id": "doc2", "text": "Doc 2", "score": 0.7},
    ]
    bm25_hits = [
        {"chunk_id": "doc2", "text": "Doc 2", "bm25_score": 5.0},
        {"chunk_id": "doc3", "text": "Doc 3", "bm25_score": 4.0},
    ]
    fused = compute_rrf_fusion(dense_hits, bm25_hits, k=60, top_k=3)
    assert len(fused) == 3
    # doc2 appeared in both lists (#2 in dense, #1 in bm25) so it should have highest RRF score
    assert fused[0]["chunk_id"] == "doc2"
    assert fused[0]["dense_rank"] == 2
    assert fused[0]["bm25_rank"] == 1


def test_mmr_diversity():
    candidates = [
        {"chunk_id": "c1", "text": "Error ERR-4032 troubleshooting step 1", "score": 0.9, "rrf_score": 0.033},
        {"chunk_id": "c2", "text": "Error ERR-4032 troubleshooting step 1 duplicate", "score": 0.85, "rrf_score": 0.032},
        {"chunk_id": "c3", "text": "SSO role mapping and IdP config guide", "score": 0.7, "rrf_score": 0.028},
    ]
    # With lambda=0.1 (high diversity penalty), c3 should be preferred over near-duplicate c2
    reranked = compute_mmr_rerank(candidates, lambda_param=0.1, top_k=2)
    assert len(reranked) == 2
    assert reranked[0]["chunk_id"] == "c1"
    assert reranked[1]["chunk_id"] == "c3"


def test_api_golden_set_and_inspection():
    client = TestClient(app)
    # Test UI root
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Customer Support RAG Workbench" in resp.text

    # Test golden set endpoint
    resp = client.get("/api/inspection/golden-set")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) == 12

    # Test compare endpoint
    resp = client.post(
        "/api/inspection/compare",
        json={"query": "ERR-4032", "top_k": 3}
    )
    assert resp.status_code == 200
    cdata = resp.json()
    assert "hybrid_chunks" in cdata
    assert len(cdata["hybrid_chunks"]) > 0


def test_api_upload_flow():
    client = TestClient(app)
    # Test uploading a sample test markdown article
    md_content = b"---\narticle_id: BM-999\nproduct_area: Test\nlast_updated: 2026-01-01\n---\n# Test Article\n\nThis is a test article for upload verification."
    files = {"files": ("test_upload.md", md_content, "text/markdown")}
    resp = client.post("/api/ingest/upload", files=files)
    assert resp.status_code == 200
    data = resp.json()
    assert "chunks_written" in data
    assert data["chunks_written"] > 0


def test_api_week3_evaluation():
    client = TestClient(app)
    resp = client.get("/api/inspection/week3/evaluation")
    assert resp.status_code == 200
    data = resp.json()
    assert "paragraph" in data
    assert "table_aware" in data
    assert "comparison" in data
    assert len(data["comparison"]) == 8
