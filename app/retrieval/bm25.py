"""BM25 lexical index, Reciprocal Rank Fusion (RRF), and MMR reranking.

Week-4 implementation:
  - BM25Okapi index with exact-token preservation for technical identifiers
    (e.g., ERR-4032, BM-002, X-Billing-Token, CUSTOM-, INV-251101-0042).
  - Reciprocal Rank Fusion (RRF) with default k=60 to fuse dense and sparse ranks
    without score normalisation artefacts.
  - Maximal Marginal Relevance (MMR) for diversity reranking on candidate pools.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any, Sequence


def tokenize(text: str) -> list[str]:
    """Tokenize text preserving hyphenated codes, numbers, and identifier tokens.

    Extracts:
      - Alphanumeric strings with internal hyphens or underscores (e.g. ERR-4032, BM-002, X-Billing-Token)
      - Standard word tokens
    """
    if not text:
        return []
    # Match words and hyphenated/underscored identifiers
    tokens = re.findall(r"[a-zA-Z0-9]+(?:[-_][a-zA-Z0-9]+)*", text.lower())
    return tokens


class BM25Index:
    """Okapi BM25 index over a collection of document chunks."""

    def __init__(
        self,
        corpus: Sequence[dict[str, Any]],
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        """
        corpus: list of dicts with at least 'chunk_id' and 'text', plus optional 'metadata'.
        """
        self.k1 = k1
        self.b = b
        self.corpus = list(corpus)
        self.doc_ids = [doc["chunk_id"] for doc in self.corpus]
        self.doc_lens = [len(tokenize(doc.get("text", ""))) for doc in self.corpus]
        self.avgdl = sum(self.doc_lens) / max(len(self.doc_lens), 1)

        self.doc_freqs: list[Counter[str]] = []
        self.df: dict[str, int] = {}
        for doc in self.corpus:
            tokens = tokenize(doc.get("text", ""))
            freqs = Counter(tokens)
            self.doc_freqs.append(freqs)
            for token in freqs:
                self.df[token] = self.df.get(token, 0) + 1

        self.N = len(self.corpus)
        self.idf: dict[str, float] = {}
        for token, freq in self.df.items():
            # Standard Lucene/Okapi smoothed IDF
            self.idf[token] = math.log((self.N - freq + 0.5) / (freq + 0.5) + 1.0)

    def search(self, query: str, top_k: int = 25) -> list[dict[str, Any]]:
        """Score all documents against query and return top_k ranked hits."""
        q_tokens = tokenize(query)
        if not q_tokens:
            return []

        scores: list[float] = []
        for i, freqs in enumerate(self.doc_freqs):
            score = 0.0
            doc_len = self.doc_lens[i]
            for token in q_tokens:
                if token in freqs:
                    tf = freqs[token]
                    idf = self.idf.get(token, 0.0)
                    denom = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avgdl))
                    numerator = tf * (self.k1 + 1.0)
                    score += idf * (numerator / denom)
            scores.append(score)

        # Pair with document info and sort
        scored_docs = []
        for doc, score in zip(self.corpus, scores):
            if score > 0.0:
                scored_docs.append({
                    "chunk_id": doc["chunk_id"],
                    "text": doc.get("text", ""),
                    "metadata": doc.get("metadata", {}),
                    "bm25_score": round(score, 4),
                })

        scored_docs.sort(key=lambda x: x["bm25_score"], reverse=True)
        return scored_docs[:top_k]


def compute_rrf_fusion(
    dense_hits: list[dict[str, Any]],
    bm25_hits: list[dict[str, Any]],
    k: int = 60,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Fuse dense and BM25 rank lists using Reciprocal Rank Fusion (RRF).

    RRF formula:
      RRF_score(d) = (1 / (k + dense_rank(d))) + (1 / (k + bm25_rank(d)))

    where ranks are 1-indexed. If a document does not appear in a rank list,
    its rank contribution for that list is 0.
    """
    doc_registry: dict[str, dict[str, Any]] = {}
    dense_ranks: dict[str, int] = {}
    bm25_ranks: dict[str, int] = {}

    for rank, hit in enumerate(dense_hits, start=1):
        cid = hit["chunk_id"]
        dense_ranks[cid] = rank
        if cid not in doc_registry:
            doc_registry[cid] = {
                "chunk_id": cid,
                "text": hit["text"],
                "metadata": hit.get("metadata", {}),
                "dense_score": hit.get("score", 0.0),
                "bm25_score": 0.0,
            }

    for rank, hit in enumerate(bm25_hits, start=1):
        cid = hit["chunk_id"]
        bm25_ranks[cid] = rank
        if cid not in doc_registry:
            doc_registry[cid] = {
                "chunk_id": cid,
                "text": hit["text"],
                "metadata": hit.get("metadata", {}),
                "dense_score": 0.0,
                "bm25_score": hit.get("bm25_score", 0.0),
            }
        else:
            doc_registry[cid]["bm25_score"] = hit.get("bm25_score", 0.0)

    fused_docs: list[dict[str, Any]] = []
    for cid, doc_info in doc_registry.items():
        dense_r = dense_ranks.get(cid)
        bm25_r = bm25_ranks.get(cid)

        rrf_score = 0.0
        if dense_r is not None:
            rrf_score += 1.0 / (k + dense_r)
        if bm25_r is not None:
            rrf_score += 1.0 / (k + bm25_r)

        fused_docs.append({
            "chunk_id": cid,
            "text": doc_info["text"],
            "metadata": doc_info["metadata"],
            "score": round(rrf_score, 6),
            "dense_score": doc_info["dense_score"],
            "bm25_score": doc_info["bm25_score"],
            "dense_rank": dense_r,
            "bm25_rank": bm25_r,
            "rrf_score": round(rrf_score, 6),
        })

    # Sort descending by RRF score
    fused_docs.sort(key=lambda x: x["rrf_score"], reverse=True)
    return fused_docs[:top_k]


def compute_mmr_rerank(
    candidates: list[dict[str, Any]],
    lambda_param: float = 0.7,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    r"""Maximal Marginal Relevance (MMR) over candidate chunk texts.

    MMR formula:
      MMR = argmax_{d in C \ S} [ lambda * Sim(q, d) - (1 - lambda) * max_{s in S} Sim(d, s) ]

    Inter-document similarity is approximated via token Jaccard similarity.
    Query relevance Sim(q, d) is normalized from candidate rank scores.
    """
    if not candidates:
        return []
    if len(candidates) <= top_k:
        return candidates

    # Pre-tokenize all candidate texts
    token_sets = [set(tokenize(c["text"])) for c in candidates]

    # Normalize relevance scores to 0..1
    raw_scores = [c.get("rrf_score", c.get("score", 1.0)) for c in candidates]
    max_sc = max(raw_scores) if raw_scores else 1.0
    min_sc = min(raw_scores) if raw_scores else 0.0
    sc_range = (max_sc - min_sc) if (max_sc - min_sc) > 1e-9 else 1.0
    norm_scores = [(s - min_sc) / sc_range for s in raw_scores]

    selected_indices: list[int] = []
    unselected = list(range(len(candidates)))

    # First item is the highest relevance candidate
    best_idx = 0
    selected_indices.append(best_idx)
    unselected.remove(best_idx)

    def jaccard(s1: set[str], s2: set[str]) -> float:
        if not s1 or not s2:
            return 0.0
        union_len = len(s1 | s2)
        return len(s1 & s2) / union_len if union_len > 0 else 0.0

    while len(selected_indices) < top_k and unselected:
        best_mmr_val = -float("inf")
        best_candidate_idx = unselected[0]

        for u in unselected:
            relevance = norm_scores[u]
            max_sim_to_selected = max(
                jaccard(token_sets[u], token_sets[s]) for s in selected_indices
            )
            mmr_score = lambda_param * relevance - (1.0 - lambda_param) * max_sim_to_selected

            if mmr_score > best_mmr_val:
                best_mmr_val = mmr_score
                best_candidate_idx = u

        selected_indices.append(best_candidate_idx)
        unselected.remove(best_candidate_idx)

    return [candidates[i] for i in selected_indices]
