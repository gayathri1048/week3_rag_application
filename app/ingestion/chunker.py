"""Turn a Ticket or Article into embeddable chunks — Strategy 1 (paragraph-pack).

Strategy 1 (this file):
  - Split on paragraph boundaries, packing paragraphs up to `chunk_size`.
  - Carry `chunk_overlap` characters of tail context into the next chunk.
  - Does NOT treat tables specially — a table may be split mid-row.

Strategy 2 (table_chunker.py):
  - Structure-aware: never separates a table row from its header row.

Week-3 additions to _metadata():
  source_file, article_id, product_area, last_updated
  (populated for Article chunks; absent for legacy Ticket chunks)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

PARAGRAPH_BREAK = re.compile(r"\n\s*\n")


@dataclass
class Chunk:
    """An embeddable unit plus the metadata Chroma stores alongside the vector."""

    chunk_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------ tickets --

def chunk_ticket(ticket: Any, chunk_size: int = 900, chunk_overlap: int = 150) -> list[Chunk]:
    """Split one Ticket into overlapping paragraph-pack chunks (Strategy 1)."""
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    body = _ticket_text(ticket)
    pieces = _split(body, chunk_size, chunk_overlap)
    header = f"Ticket {ticket.id} — {ticket.subject}"

    return [
        Chunk(
            chunk_id=f"{ticket.id}::p{index}",
            text=f"{header}\n\n{piece}",
            metadata=_ticket_metadata(ticket, index, len(pieces)),
        )
        for index, piece in enumerate(pieces)
    ]


def _ticket_text(ticket: Any) -> str:
    sections = [ticket.body.strip()]
    if ticket.resolution:
        sections.append(f"Resolution: {ticket.resolution.strip()}")
    if ticket.tags:
        sections.append(f"Tags: {', '.join(ticket.tags)}")
    return "\n\n".join(s for s in sections if s)


def _ticket_metadata(ticket: Any, index: int, total: int) -> dict[str, Any]:
    """Chroma metadata values must be str/int/float/bool — never None or list."""
    meta: dict[str, Any] = {
        "doc_type": "ticket",
        "ticket_id": ticket.id,
        "subject": ticket.subject,
        "category": ticket.category.value,
        "priority": ticket.priority.value,
        "status": ticket.status.value,
        "chunk_index": index,
        "chunk_total": total,
        "has_resolution": bool(ticket.resolution),
        "chunker": "paragraph",           # strategy tag for comparison queries
    }
    if ticket.product:
        meta["product"] = ticket.product
    if ticket.customer:
        meta["customer"] = ticket.customer
    if ticket.created_at:
        meta["created_at"] = ticket.created_at.isoformat()
    if ticket.tags:
        meta["tags"] = ", ".join(ticket.tags)
    return meta


# ------------------------------------------------------------------ articles --

def chunk_article(article: Any, chunk_size: int = 900, chunk_overlap: int = 150) -> list[Chunk]:
    """Split one Article into overlapping paragraph-pack chunks (Strategy 1).

    Week-3 required metadata fields are stamped on every chunk:
      source_file, article_id, product_area, last_updated
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    pieces = _split(article.body.strip(), chunk_size, chunk_overlap)
    header = f"Article {article.article_id} — {article.title}"

    return [
        Chunk(
            chunk_id=f"{article.article_id}::p{index}",
            text=f"{header}\n\n{piece}",
            metadata=_article_metadata(article, index, len(pieces)),
        )
        for index, piece in enumerate(pieces)
    ]


def _article_metadata(article: Any, index: int, total: int) -> dict[str, Any]:
    """Week-3 required metadata fields on every article chunk."""
    meta: dict[str, Any] = {
        "doc_type": "article",
        # --- Week-3 required fields ---
        "source_file": article.source_file,
        "article_id": article.article_id,
        "product_area": article.product_area,
        "last_updated": str(article.last_updated),
        # --- housekeeping ---
        "subject": article.title,
        "chunk_index": index,
        "chunk_total": total,
        "chunker": "paragraph",           # strategy tag
    }
    if article.tags:
        meta["tags"] = ", ".join(article.tags)
    return meta


# --------------------------------------------------------------- shared split --

def _split(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Pack paragraphs into chunks, hard-splitting any paragraph that is itself too long."""
    if not text:
        return [""]

    paragraphs: list[str] = []
    for para in PARAGRAPH_BREAK.split(text):
        para = para.strip()
        if not para:
            continue
        if len(para) <= chunk_size:
            paragraphs.append(para)
        else:
            paragraphs.extend(_hard_split(para, chunk_size))

    chunks: list[str] = []
    current = ""
    for para in paragraphs:
        candidate = f"{current}\n\n{para}" if current else para
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
            tail = current[-chunk_overlap:] if chunk_overlap else ""
            current = f"{tail}\n\n{para}".strip() if tail else para
        else:
            current = para

    if current:
        chunks.append(current)
    return chunks or [""]


def _hard_split(para: str, chunk_size: int) -> list[str]:
    """Break an oversized paragraph on sentence boundaries, falling back to slices."""
    sentences = re.split(r"(?<=[.!?])\s+", para)
    out: list[str] = []
    current = ""
    for sentence in sentences:
        while len(sentence) > chunk_size:
            out.append(sentence[:chunk_size])
            sentence = sentence[chunk_size:]
        candidate = f"{current} {sentence}".strip()
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                out.append(current)
            current = sentence
    if current:
        out.append(current)
    return out
