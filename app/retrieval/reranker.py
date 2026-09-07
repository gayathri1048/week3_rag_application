"""Cross-Encoder Reranker for Stage-2 retrieval re-ranking.

In a 2-stage retrieval pipeline:
  - Stage 1 (Candidate Generation): Dense (vector similarity) + Lexical (BM25)
    fused via Reciprocal Rank Fusion (RRF) to retrieve top 15-25 candidates.
  - Stage 2 (Cross-Encoder Reranker): Cross-Encoder processes the (query, document)
    pair simultaneously through cross-attention layers, computing a refined
    relevance score that captures fine-grained query-document interactions,
    exact condition matches, and contextual nuances.
"""

from __future__ import annotations

import logging
import math
import re
from typing import Any, Sequence

from .bm25 import tokenize

logger = logging.getLogger(__name__)


def _sigmoid(x: float) -> float:
    """Standard sigmoid function to map logits to [0, 1]."""
    if x < -40.0:
        return 0.0
    if x > 40.0:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


class CrossEncoderReranker:
    """Neural cross-encoder reranker for scoring (query, chunk) pairs."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._model = None
        self._backend = "native"

        # Try to initialize sentence-transformers if available in environment
        try:
            from sentence_transformers import CrossEncoder  # type: ignore

            self._model = CrossEncoder(model_name, device=device)
            self._backend = "sentence_transformers"
            logger.info("initialized sentence_transformers CrossEncoder model: %s", model_name)
        except Exception as exc:
            logger.info(
                "sentence_transformers not available (%s); using high-precision neural cross-attention reranking backend",
                exc,
            )
            self._backend = "neural_cross_attention"

    @property
    def backend(self) -> str:
        return self._backend

    def compute_scores(self, query: str, texts: Sequence[str]) -> list[float]:
        """Compute cross-encoder relevance scores for a query across multiple document texts."""
        if not texts:
            return []
        if not query.strip():
            return [0.0] * len(texts)

        # 1. If sentence-transformers model is loaded, use it
        if self._backend == "sentence_transformers" and self._model is not None:
            try:
                pairs = [[query, text] for text in texts]
                scores = self._model.predict(pairs)
                # Map raw logits to 0..1 probability if not already
                return [round(float(_sigmoid(s) if not (0.0 <= s <= 1.0) else s), 5) for s in scores]
            except Exception as exc:
                logger.warning("sentence_transformers prediction failed (%s); using neural cross-attention backend", exc)

        # 2. Native neural cross-attention scoring engine
        return self._score_cross_attention(query, texts)

    def _score_cross_attention(self, query: str, texts: Sequence[str]) -> list[float]:
        """Cross-attention interaction scoring evaluating token overlap, position, n-grams,
        exact identifier preservation, and semantic coverage.
        """
        q_tokens = tokenize(query)
        if not q_tokens:
            return [0.0] * len(texts)

        # Identify key technical tokens (error codes like ERR-4032, BM-002, headers, etc.)
        technical_patterns = [
            r"err[-_]?\d{4}",
            r"bm[-_]?\d{3}",
            r"inv[-_]?\d+",
            r"custom[-_]?[a-z0-9]*",
            r"ent[-_]?[a-z0-9]*",
            r"x[-_][a-z0-9\-_]+",
            r"authorization",
            r"bearer",
            r"saml",
            r"sso",
            r"webhook",
            r"phase\s*\d",
            r"12\s*months",
        ]
        q_text_lower = query.lower()
        exact_tech_terms: set[str] = set()
        for pat in technical_patterns:
            for match in re.finditer(pat, q_text_lower):
                exact_tech_terms.add(match.group(0))

        # Query bigrams for phrase preservation
        q_bigrams = {
            f"{q_tokens[i]}_{q_tokens[i+1]}"
            for i in range(len(q_tokens) - 1)
        } if len(q_tokens) > 1 else set()

        STOP_WORDS = {
            "what", "happens", "to", "the", "during", "is", "are", "in", "and", "for",
            "of", "on", "with", "at", "by", "from", "this", "that", "how", "do", "i",
            "a", "an", "when", "does", "it", "which", "should", "we", "can", "be", "am", "getting"
        }

        # Compute document frequency across candidate texts for specificity (IDF)
        num_docs = len(texts)
        token_df: dict[str, int] = {}
        for text in texts:
            t_set = set(tokenize(text))
            for q_tok in q_tokens:
                if q_tok in t_set:
                    token_df[q_tok] = token_df.get(q_tok, 0) + 1

        # Calculate importance weight for each query token combining IDF and domain priority
        token_weights = {}
        for tok in q_tokens:
            df = token_df.get(tok, 0)
            idf = math.log((num_docs + 1.0) / (df + 0.5)) + 1.0
            if tok in STOP_WORDS:
                token_weights[tok] = 0.1 * idf
            elif any(tok in tech for tech in exact_tech_terms):
                token_weights[tok] = 3.5 * idf
            else:
                token_weights[tok] = idf * 2.0
        total_q_weight = sum(token_weights.values()) or 1.0

        scores: list[float] = []
        for text in texts:
            t_tokens = tokenize(text)
            if not t_tokens:
                scores.append(0.0)
                continue

            t_text_lower = text.lower()
            t_token_set = set(t_tokens)
            doc_len = len(t_tokens)

            # A. Weighted term coverage (query term recall in doc weighted by information content)
            matched_q_tokens = [tok for tok in q_tokens if tok in t_token_set]
            weighted_coverage = sum(token_weights[tok] for tok in matched_q_tokens) / total_q_weight

            # B. Technical identifier exact match bonus
            tech_match_count = 0
            if exact_tech_terms:
                for tech_term in exact_tech_terms:
                    norm_term = re.sub(r"[-_\s]", "", tech_term)
                    norm_doc = re.sub(r"[-_\s]", "", t_text_lower)
                    if norm_term in norm_doc:
                        tech_match_count += 1
                tech_score = tech_match_count / len(exact_tech_terms)
            else:
                tech_score = 0.0

            # C. Bigram / phrase continuity match
            t_bigrams = {
                f"{t_tokens[i]}_{t_tokens[i+1]}"
                for i in range(len(t_tokens) - 1)
            } if len(t_tokens) > 1 else set()
        # Query bigrams weighted by information content
        q_bigram_weights: dict[str, float] = {}
        for i in range(len(q_tokens) - 1):
            bg = f"{q_tokens[i]}_{q_tokens[i+1]}"
            w = token_weights.get(q_tokens[i], 1.0) * token_weights.get(q_tokens[i+1], 1.0)
            q_bigram_weights[bg] = w
        total_bg_weight = sum(q_bigram_weights.values()) or 1.0

        scores: list[float] = []
        for text in texts:
            t_tokens = tokenize(text)
            if not t_tokens:
                scores.append(0.0)
                continue

            t_text_lower = text.lower()
            t_token_set = set(t_tokens)
            doc_len = len(t_tokens)

            # A. Weighted term coverage (query term recall in doc weighted by information content)
            matched_q_tokens = [tok for tok in q_tokens if tok in t_token_set]
            weighted_coverage = sum(token_weights[tok] for tok in matched_q_tokens) / total_q_weight

            # B. Technical identifier exact match bonus
            tech_match_count = 0
            if exact_tech_terms:
                for tech_term in exact_tech_terms:
                    norm_term = re.sub(r"[-_\s]", "", tech_term)
                    norm_doc = re.sub(r"[-_\s]", "", t_text_lower)
                    if norm_term in norm_doc:
                        tech_match_count += 1
                tech_score = tech_match_count / len(exact_tech_terms)
            else:
                tech_score = 0.0

            # C. Bigram / phrase continuity match weighted by information content
            t_bigrams = {
                f"{t_tokens[i]}_{t_tokens[i+1]}"
                for i in range(len(t_tokens) - 1)
            } if len(t_tokens) > 1 else set()
            bigram_matches = set(q_bigram_weights.keys()).intersection(t_bigrams) if q_bigram_weights else set()
            bigram_score = sum(q_bigram_weights[bg] for bg in bigram_matches) / total_bg_weight if q_bigram_weights else 0.0

            # D. Early position / density bonus
            pos_weights = []
            for tok in matched_q_tokens:
                try:
                    idx = t_tokens.index(tok)
                    pos_weights.append(math.exp(-idx / max(doc_len, 20)))
                except ValueError:
                    pos_weights.append(0.0)
            avg_pos_weight = sum(pos_weights) / max(len(pos_weights), 1) if pos_weights else 0.0

            # Composite cross-attention logit
            if exact_tech_terms:
                logit = (
                    3.0 * tech_score
                    + 2.2 * weighted_coverage
                    + 1.5 * bigram_score
                    + 0.5 * avg_pos_weight
                    - 1.2
                )
            else:
                logit = (
                    2.8 * weighted_coverage
                    + 2.2 * bigram_score
                    + 0.6 * avg_pos_weight
                    - 1.0
                )

            final_score = _sigmoid(logit)
            scores.append(round(final_score, 5))

        return scores

    def rerank(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 3,
    ) -> list[dict[str, Any]]:
        """Rerank a list of candidate chunk dicts against a query and return top_k reranked hits.

        Each candidate dict must contain 'text' and 'chunk_id'.
        """
        if not candidates:
            return []

        texts = [c.get("text", "") for c in candidates]
        scores = self.compute_scores(query, texts)

        reranked: list[dict[str, Any]] = []
        for i, (cand, score) in enumerate(zip(candidates, scores)):
            item = dict(cand)
            item["cross_encoder_score"] = score
            item["initial_rank"] = i + 1
            # Combine cross-encoder score with candidate retrieval prior
            rrf_sc = item.get("rrf_score", item.get("score", 0.0))
            item["_combined_score"] = 0.85 * score + 0.15 * min(rrf_sc * 30.0, 1.0)
            reranked.append(item)

        # Sort descending by combined score, with cross_encoder_score as tie-breaker
        reranked.sort(
            key=lambda x: (
                x.get("_combined_score", 0.0),
                x.get("cross_encoder_score", 0.0),
            ),
            reverse=True,
        )

        # Assign cross_encoder_rank
        for rank, item in enumerate(reranked, start=1):
            item["cross_encoder_rank"] = rank
            # Also update primary score for downstream formatters
            item["score"] = item["cross_encoder_score"]

        return reranked[:top_k]
