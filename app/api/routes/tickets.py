"""Ticket browsing and collection statistics, served from the vector store metadata."""

from __future__ import annotations

import logging
from collections import Counter

from fastapi import APIRouter, Depends, HTTPException, Query, status

from ...dependencies import get_vector_store
from ...retrieval.vector_store import VectorStore
from ...schemas import (
    Category,
    CollectionStats,
    Priority,
    Status,
    Ticket,
    TicketListResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/tickets", response_model=TicketListResponse)
def list_tickets(
    store: VectorStore = Depends(get_vector_store),
    category: Category | None = None,
    priority: Priority | None = None,
    ticket_status: Status | None = Query(default=None, alias="status"),
    search: str | None = Query(default=None, description="Case-insensitive substring match."),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> TicketListResponse:
    """List indexed tickets.

    Chroma stores chunks, not tickets, so chunks are reassembled by `ticket_id`
    before filtering and paging.
    """
    tickets = _reassemble(store)

    if category:
        tickets = [t for t in tickets if t.category == category]
    if priority:
        tickets = [t for t in tickets if t.priority == priority]
    if ticket_status:
        tickets = [t for t in tickets if t.status == ticket_status]
    if search:
        needle = search.lower()
        tickets = [
            t for t in tickets if needle in t.subject.lower() or needle in t.body.lower()
        ]

    return TicketListResponse(
        tickets=tickets[offset : offset + limit],
        total=len(tickets),
    )


@router.get("/tickets/{ticket_id}", response_model=Ticket)
def get_ticket(ticket_id: str, store: VectorStore = Depends(get_vector_store)) -> Ticket:
    """Fetch one reassembled ticket."""
    for ticket in _reassemble(store):
        if ticket.id == ticket_id:
            return ticket
    raise HTTPException(status.HTTP_404_NOT_FOUND, f"Ticket {ticket_id!r} is not indexed.")


@router.get("/stats", response_model=CollectionStats)
def stats(store: VectorStore = Depends(get_vector_store)) -> CollectionStats:
    """Counts by category, priority, and status — backs the dashboard."""
    tickets = _reassemble(store)
    return CollectionStats(
        collection=store.collection_name,
        chunk_count=store.count(),
        ticket_count=len(tickets),
        by_category=dict(Counter(t.category.value for t in tickets)),
        by_priority=dict(Counter(t.priority.value for t in tickets)),
        by_status=dict(Counter(t.status.value for t in tickets)),
    )


def _reassemble(store: VectorStore) -> list[Ticket]:
    """Rebuild Ticket objects from stored chunks, ordered by chunk_index.

    Each chunk text carries a `Ticket <id> — <subject>` header that was added for
    retrieval quality; it is stripped here so the UI shows the original body.
    """
    by_ticket: dict[str, list[dict]] = {}
    for record in store.get_all():
        meta = record["metadata"]
        ticket_id = str(meta.get("ticket_id") or record["chunk_id"].split("::")[0])
        by_ticket.setdefault(ticket_id, []).append(record)

    tickets: list[Ticket] = []
    for ticket_id, records in by_ticket.items():
        records.sort(key=lambda r: int(r["metadata"].get("chunk_index", 0)))
        meta = records[0]["metadata"]
        body = "\n\n".join(_strip_header(r["text"]) for r in records)

        tags = meta.get("tags")
        try:
            tickets.append(
                Ticket(
                    id=ticket_id,
                    subject=str(meta.get("subject", "(untitled)")),
                    body=body,
                    customer=meta.get("customer"),
                    category=meta.get("category", Category.OTHER),
                    priority=meta.get("priority", Priority.MEDIUM),
                    status=meta.get("status", Status.OPEN),
                    product=meta.get("product"),
                    created_at=meta.get("created_at"),
                    tags=[t.strip() for t in tags.split(",")] if isinstance(tags, str) else [],
                )
            )
        except Exception:
            logger.warning("could not reassemble ticket %s", ticket_id, exc_info=True)

    tickets.sort(key=lambda t: (t.created_at is None, t.created_at), reverse=True)
    return tickets


def _strip_header(text: str) -> str:
    """Drop the `Ticket <id> — <subject>` line the chunker prepends."""
    _, separator, rest = text.partition("\n\n")
    return (rest if separator else text).strip()
