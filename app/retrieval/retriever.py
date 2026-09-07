"""Query-time retrieval: filter, search, threshold, de-duplicate, format for the prompt.

Week-4 additions:
  - Hybrid retrieval: BM25 + Reciprocal Rank Fusion (RRF, k=60)
  - Mode selection: 'hybrid' (default), 'dense' (baseline), 'bm25', 'mmr'
  - Maximal Marginal Relevance (MMR) diversity reranking for bonus challenge
  - Rich diagnostics: dense_rank, bm25_rank, dense_score, bm25_score, rrf_score
"""

from __future__ import annotations

import logging
from typing import Any, Literal

from ..config import get_settings
from ..schemas import (
    ArticleFilters,
    Category,
    Priority,
    RetrievedChunk,
    Status,
    TicketFilters,
)
from .bm25 import BM25Index, compute_mmr_rerank, compute_rrf_fusion
from .reranker import CrossEncoderReranker
from .vector_store import VectorStore

logger = logging.getLogger(__name__)

OVERFETCH_FACTOR = 3


class Retriever:
    """Turns a natural-language question into a ranked list of chunks."""

    def __init__(
        self,
        store: VectorStore,
        top_k: int = 3,
        min_relevance: float = 0.0,
        default_mode: Literal["hybrid", "dense", "bm25", "mmr", "rerank"] = "hybrid",
        rrf_k: int = 60,
        reranker: CrossEncoderReranker | None = None,
    ) -> None:
        self.store = store
        self.top_k = top_k
        self.min_relevance = min_relevance
        self.default_mode = default_mode
        self.rrf_k = rrf_k
        self._bm25_index: BM25Index | None = None
        self._reranker: CrossEncoderReranker | None = reranker

    def _get_bm25_index(self) -> BM25Index:
        """Lazily build or refresh the BM25 index over vector store documents."""
        if self._bm25_index is None:
            raw_docs = self.store.get_all()
            self._bm25_index = BM25Index(raw_docs)
        return self._bm25_index

    def _get_reranker(self) -> CrossEncoderReranker:
        """Lazily build or return the Cross-Encoder reranker."""
        if self._reranker is None:
            settings = get_settings()
            self._reranker = CrossEncoderReranker(model_name=settings.reranker_model)
        return self._reranker

    def reset_bm25_index(self) -> None:
        """Clear cached BM25 index (e.g. after ingestion/reset)."""
        self._bm25_index = None

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: TicketFilters | None = None,
        article_filters: ArticleFilters | None = None,
        mode: Literal["hybrid", "dense", "bm25", "mmr", "rerank"] | None = None,
        max_per_source: int | None = None,
        mmr_lambda: float = 0.7,
        rrf_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve the most relevant chunks for `query`.

        Modes:
          - 'dense': Dense cosine similarity from Chroma (Week-3 baseline)
          - 'bm25': Lexical Okapi BM25 ranking
          - 'hybrid': BM25 + Dense fused via Reciprocal Rank Fusion (RRF, k=60)
          - 'mmr': Maximal Marginal Relevance over hybrid candidates
          - 'rerank': 2-stage retrieval: Hybrid RRF candidate generation + Cross-Encoder re-ranking
        """
        limit = top_k or self.top_k
        active_mode = mode or self.default_mode
        k_const = rrf_k or self.rrf_k
        where = _build_where(filters, article_filters)

        # 1. Dense retrieval
        dense_fetch_k = max(limit * OVERFETCH_FACTOR, 25)
        dense_hits = self.store.query(query, top_k=dense_fetch_k, where=where)

        if active_mode == "dense":
            if not dense_hits:
                logger.info("no dense hits for query=%r", query[:80])
                return []
            chunks = []
            for rank, hit in enumerate(dense_hits, start=1):
                c = _to_chunk(hit)
                c.dense_rank = rank
                c.dense_score = round(hit["score"], 4)
                c.retrieval_mode = "dense"
                chunks.append(c)

            chunks = [c for c in chunks if c.score >= self.min_relevance]
            return _apply_dedup_and_limit(chunks, limit, max_per_source)

        # 2. BM25 retrieval
        bm25_index = self._get_bm25_index()
        bm25_hits = bm25_index.search(query, top_k=25)

        # Filter BM25 hits by metadata where clause if provided
        if where:
            bm25_hits = [h for h in bm25_hits if _matches_where(h.get("metadata", {}), where)]

        if active_mode == "bm25":
            if not bm25_hits:
                logger.info("no bm25 hits for query=%r", query[:80])
                return []
            chunks = []
            for rank, hit in enumerate(bm25_hits, start=1):
                c = _to_chunk(hit)
                c.bm25_rank = rank
                c.bm25_score = hit["bm25_score"]
                c.score = hit["bm25_score"]
                c.retrieval_mode = "bm25"
                chunks.append(c)
            return _apply_dedup_and_limit(chunks, limit, max_per_source)

        # 3. Hybrid RRF Fusion (BM25 + Dense)
        candidate_pool_size = max(limit * 4, 25) if active_mode == "rerank" else max(limit * 2, 25)
        fused_hits = compute_rrf_fusion(
            dense_hits=dense_hits,
            bm25_hits=bm25_hits,
            k=k_const,
            top_k=candidate_pool_size,
        )

        if not fused_hits:
            logger.info("no fused hits for query=%r", query[:80])
            return []

        if active_mode == "mmr":
            fused_hits = compute_mmr_rerank(
                candidates=fused_hits,
                lambda_param=mmr_lambda,
                top_k=limit,
            )
        elif active_mode == "rerank":
            reranker = self._get_reranker()
            fused_hits = reranker.rerank(
                query=query,
                candidates=fused_hits,
                top_k=limit,
            )

        chunks = []
        for hit in fused_hits:
            c = _to_chunk(hit)
            c.dense_rank = hit.get("dense_rank")
            c.bm25_rank = hit.get("bm25_rank")
            c.dense_score = hit.get("dense_score")
            c.bm25_score = hit.get("bm25_score")
            c.rrf_score = hit.get("rrf_score")
            c.cross_encoder_score = hit.get("cross_encoder_score")
            c.cross_encoder_rank = hit.get("cross_encoder_rank")
            c.score = hit.get("cross_encoder_score", hit.get("rrf_score", hit.get("score", 0.0)))
            c.retrieval_mode = active_mode
            chunks.append(c)

        return _apply_dedup_and_limit(chunks, limit, max_per_source)

    def similar_to_ticket(
        self, subject: str, body: str, top_k: int = 4, resolved_only: bool = True
    ) -> list[RetrievedChunk]:
        """Find past tickets like this one — backs triage and reply drafting."""
        where: dict[str, Any] | None = {"has_resolution": True} if resolved_only else None
        hits = self.store.query(f"{subject}\n\n{body}", top_k=top_k * OVERFETCH_FACTOR, where=where)

        seen: set[str] = set()
        out: list[RetrievedChunk] = []
        for chunk in (_to_chunk(hit) for hit in hits):
            if chunk.ticket_id in seen:
                continue
            seen.add(chunk.ticket_id)
            out.append(chunk)
            if len(out) >= top_k:
                break
        return out


def _apply_dedup_and_limit(
    chunks: list[RetrievedChunk],
    limit: int,
    max_per_source: int | None = None,
) -> list[RetrievedChunk]:
    """Optionally limit chunks per source file/article/ticket, then cap at limit."""
    if max_per_source is None:
        return chunks[:limit]

    per_source: dict[str, int] = {}
    selected: list[RetrievedChunk] = []
    for chunk in chunks:
        source_key = chunk.article_id or chunk.ticket_id
        seen = per_source.get(source_key, 0)
        if seen >= max_per_source:
            continue
        per_source[source_key] = seen + 1
        selected.append(chunk)
        if len(selected) >= limit:
            break
    return selected


def _matches_where(meta: dict[str, Any], where: dict[str, Any]) -> bool:
    """Evaluate simple where conditions against a chunk's metadata."""
    if not where:
        return True
    if "$and" in where:
        return all(_matches_where(meta, clause) for clause in where["$and"])
    for k, v in where.items():
        if not k.startswith("$"):
            if meta.get(k) != v:
                return False
    return True


def _build_where(
    filters: TicketFilters | None,
    article_filters: ArticleFilters | None,
) -> dict[str, Any] | None:
    """Translate filters into a Chroma `where` clause."""
    clauses: list[dict[str, Any]] = []

    if filters:
        for field, value in (
            ("category", filters.category),
            ("priority", filters.priority),
            ("status", filters.status),
            ("product", filters.product),
        ):
            if value is not None:
                clauses.append({field: value.value if hasattr(value, "value") else value})

    if article_filters:
        if article_filters.product_area is not None:
            clauses.append({"product_area": article_filters.product_area})
        if article_filters.article_id is not None:
            clauses.append({"article_id": article_filters.article_id})

    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def _to_chunk(hit: dict[str, Any]) -> RetrievedChunk:
    meta = hit.get("metadata") or {}
    score = hit.get("score", hit.get("rrf_score", 0.0))
    return RetrievedChunk(
        chunk_id=hit["chunk_id"],
        ticket_id=str(meta.get("ticket_id", meta.get("article_id", hit["chunk_id"].split("::")[0]))),
        subject=str(meta.get("subject", "(untitled)")),
        text=hit.get("text", ""),
        score=round(float(score), 6),
        # ticket fields
        category=_enum(Category, meta.get("category")),
        priority=_enum(Priority, meta.get("priority")),
        status=_enum(Status, meta.get("status")),
        product=meta.get("product"),
        # Week-3 article fields
        source_file=meta.get("source_file"),
        article_id=meta.get("article_id"),
        product_area=meta.get("product_area"),
        last_updated=meta.get("last_updated"),
    )


def _enum(enum_cls: type, value: Any) -> Any:
    """Coerce a stored string back to its enum, tolerating unknown values."""
    if value is None:
        return None
    try:
        return enum_cls(value)
    except ValueError:
        return None


def format_context(chunks: list[RetrievedChunk]) -> str:
    """Render chunks as the <source> block the model sees."""
    if not chunks:
        return "(no matching sources found in the knowledge base)"

    blocks = []
    for index, chunk in enumerate(chunks, start=1):
        attrs: list[str] = [
            f'index="{index}"',
            f'chunk_id="{chunk.chunk_id}"',
        ]
        if chunk.article_id:
            attrs.append(f'article_id="{chunk.article_id}"')
        if chunk.source_file:
            attrs.append(f'source_file="{chunk.source_file}"')
        if chunk.product_area:
            attrs.append(f'product_area="{chunk.product_area}"')
        if chunk.last_updated:
            attrs.append(f'last_updated="{chunk.last_updated}"')
        for name in ("category", "priority", "status", "product"):
            val = getattr(chunk, name)
            if val is not None:
                attrs.append(f'{name}="{val.value if hasattr(val, "value") else val}"')
        attrs.append(f'relevance="{chunk.score:.4f}"')

        blocks.append(
            f'<source {" ".join(attrs)}>\n'
            f"{chunk.text.strip()}\n"
            f"</source>"
        )
    return "\n\n".join(blocks)
