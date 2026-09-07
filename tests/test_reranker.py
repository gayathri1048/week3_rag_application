"""Unit and integration tests for Cross-Encoder Reranker."""

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.retrieval.reranker import CrossEncoderReranker
from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import VectorStore
from app.config import get_settings


def test_cross_encoder_scoring():
    reranker = CrossEncoderReranker()
    query = "Customer hit ERR-4032 during subscription migration"
    texts = [
        "General billing contact details and business hours.",
        "Error ERR-4032 indicates subscription missing from UBP catalog. Fix: re-sync catalog in admin.",
        "How to configure SAML SSO attributes.",
    ]
    scores = reranker.compute_scores(query, texts)
    assert len(scores) == 3
    # The second text contains exact ERR-4032 and relevant terms, so it must score highest
    assert scores[1] > scores[0]
    assert scores[1] > scores[2]


def test_cross_encoder_rerank_order():
    reranker = CrossEncoderReranker()
    query = "What header replaces X-Billing-Token in API requests?"
    candidates = [
        {"chunk_id": "c1", "text": "Invoice PDF downloads are available under the Invoices tab."},
        {"chunk_id": "c2", "text": "All API requests now require Authorization: Bearer <token> replacing legacy X-Billing-Token."},
        {"chunk_id": "c3", "text": "Credit expiration happens 12 months after grant date."},
    ]
    reranked = reranker.rerank(query, candidates, top_k=2)
    assert len(reranked) == 2
    assert reranked[0]["chunk_id"] == "c2"
    assert reranked[0]["cross_encoder_rank"] == 1
    assert "cross_encoder_score" in reranked[0]


def test_retriever_rerank_mode():
    settings = get_settings()
    store = VectorStore(settings.chroma_path, "support_articles_table_aware")
    retriever = Retriever(store, top_k=3, min_relevance=0.0)

    chunks = retriever.retrieve("ERR-4032", top_k=3, mode="rerank")
    assert len(chunks) > 0
    assert chunks[0].retrieval_mode == "rerank"
    assert chunks[0].cross_encoder_score is not None
    assert chunks[0].cross_encoder_rank == 1


def test_api_compare_includes_rerank():
    client = TestClient(app)
    resp = client.post(
        "/api/inspection/compare",
        json={"query": "ERR-4032 subscription migration error", "top_k": 3}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "dense_chunks" in data
    assert "bm25_chunks" in data
    assert "hybrid_chunks" in data
    assert "rerank_chunks" in data
    assert len(data["rerank_chunks"]) > 0


def test_api_direct_rerank_endpoint():
    client = TestClient(app)
    resp = client.post(
        "/api/inspection/rerank",
        json={
            "query": "ERR-4001 card re-tokenisation",
            "passages": [
                "Unrelated passage about SSO and user roles.",
                "ERR-4001 indicates saved card token could not be re-tokenised because card expired.",
            ],
            "top_k": 2
        }
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["query"] == "ERR-4001 card re-tokenisation"
    assert len(data["results"]) == 2
    assert data["results"][0]["rank"] == 1
    assert "ERR-4001" in data["results"][0]["text"]


def test_api_chat_with_rerank_mode():
    client = TestClient(app)
    resp = client.post(
        "/api/chat",
        json={
            "message": "What is the expiration timeframe for unused migration credits?",
            "mode": "rerank",
            "top_k": 3,
        }
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "answer" in data
    assert "sources" in data
    assert len(data["sources"]) > 0
    assert data["sources"][0]["cross_encoder_score"] is not None
