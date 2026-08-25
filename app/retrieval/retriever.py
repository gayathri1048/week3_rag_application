"""Query-time retrieval: filter, search, threshold, de-duplicate, format for the prompt.

Week-3 additions:
  - product_area filter (new metadata field on article chunks)
  - article_id filter
  - _to_chunk() now populates source_file, article_id, product_area, last_updated
  - format_context() exposes chunk_id in the prompt so the model can cite it directly
"""

from __future__ import annotations

import logging
from typing import Any

from ..schemas import (
    ArticleFilters,
    Category,
    Priority,
    RetrievedChunk,
    Status,
    TicketFilters,
)
from .vector_store import VectorStore

logger = logging.getLogger(__name__)

OVERFETCH_FACTOR = 3


class Retriever:
    """Turns a natural-language question into a ranked list of chunks."""

    def __init__(self, store: VectorStore, top_k: int = 6, min_relevance: float = 0.25) -> None:
        self.store = store
        self.top_k = top_k
        self.min_relevance = min_relevance

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filters: TicketFilters | None = None,
        article_filters: ArticleFilters | None = None,
        max_per_source: int = 2,
    ) -> list[RetrievedChunk]:
        """Retrieve the most relevant chunks for `query`."""
        limit = top_k or self.top_k
        where = _build_where(filters, article_filters)

        hits = self.store.query(query, top_k=limit * OVERFETCH_FACTOR, where=where)
        if not hits:
            logger.info("no hits for query=%r filters=%s", query[:80], where)
            return []

        chunks = [_to_chunk(hit) for hit in hits]
        chunks = [c for c in chunks if c.score >= self.min_relevance]

        # De-duplicate: first-seen wins the per-source cap
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

        logger.info(
            "retrieved %d/%d chunks (threshold=%.2f) for query=%r",
            len(selected), len(hits), self.min_relevance, query[:80],
        )
        return selected

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


def _build_where(
    filters: TicketFilters | None,
    article_filters: ArticleFilters | None,
) -> dict[str, Any] | None:
    """Translate filters into a Chroma `where` clause.

    Week-3: product_area and article_id now supported as filter fields.
    """
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
    meta = hit["metadata"]
    return RetrievedChunk(
        chunk_id=hit["chunk_id"],
        ticket_id=str(meta.get("ticket_id", meta.get("article_id", hit["chunk_id"].split("::")[0]))),
        subject=str(meta.get("subject", "(untitled)")),
        text=hit["text"],
        score=round(hit["score"], 4),
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
    """Render chunks as the <source> block the model sees.

    Week-3: chunk_id is now included in the XML tag so the model can cite it
    directly (e.g. [chunk_id: BM-002::t3]) rather than a positional [1] index
    that cannot be resolved after the response is generated.
    """
    if not chunks:
        return "(no matching sources found in the knowledge base)"

    blocks = []
    for index, chunk in enumerate(chunks, start=1):
        # Build attribute string
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
        attrs.append(f'relevance="{chunk.score:.2f}"')

        blocks.append(
            f'<source {" ".join(attrs)}>\n'
            f"{chunk.text.strip()}\n"
            f"</source>"
        )
    return "\n\n".join(blocks)
