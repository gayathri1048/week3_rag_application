"""Pydantic models — the wire contract shared with the frontend.

Week-3 additions:
  - Article model (help-centre articles, distinct from Ticket)
  - RetrievedChunk extended with source_file, article_id, product_area, last_updated
  - ArticleFilters with product_area for metadata filtering
"""

from datetime import date, datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Domain enums
# --------------------------------------------------------------------------- #
class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"


class Status(str, Enum):
    OPEN = "open"
    PENDING = "pending"
    RESOLVED = "resolved"
    CLOSED = "closed"


class Category(str, Enum):
    BILLING = "billing"
    TECHNICAL = "technical"
    ACCOUNT = "account"
    SHIPPING = "shipping"
    PRODUCT = "product"
    OTHER = "other"


class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    FRUSTRATED = "frustrated"
    ANGRY = "angry"


# --------------------------------------------------------------------------- #
# Tickets (unchanged from Week-2 baseline)
# --------------------------------------------------------------------------- #
class Ticket(BaseModel):
    """A support ticket as stored in `documents/` and indexed into Chroma."""

    id: str
    subject: str
    body: str
    customer: str | None = None
    category: Category = Category.OTHER
    priority: Priority = Priority.MEDIUM
    status: Status = Status.OPEN
    product: str | None = None
    resolution: str | None = None
    created_at: datetime | None = None
    tags: list[str] = Field(default_factory=list)


class TicketFilters(BaseModel):
    """Metadata pre-filters applied inside the vector search."""

    category: Category | None = None
    priority: Priority | None = None
    status: Status | None = None
    product: str | None = None


class TicketListResponse(BaseModel):
    tickets: list[Ticket]
    total: int


# --------------------------------------------------------------------------- #
# Articles  ← NEW for Week 3
# --------------------------------------------------------------------------- #
class Article(BaseModel):
    """A help-centre article ingested for the Week-3 billing-migration drop.

    The four fields below are mandatory per the task spec:
      source_file  — filename (e.g. billing-migration-01.md)
      article_id   — stable slug (e.g. BM-001)
      product_area — coarse product grouping used for metadata filtering
      last_updated — ISO date of the article revision
    """

    article_id: str                        # BM-001 … BM-006
    title: str
    body: str                              # full markdown body
    source_file: str                       # filename, populated by loader
    product_area: str                      # e.g. "Billing", "Payments"
    last_updated: date
    tags: list[str] = Field(default_factory=list)


class ArticleFilters(BaseModel):
    """Metadata pre-filters for article-only searches."""

    product_area: str | None = None
    article_id: str | None = None


# --------------------------------------------------------------------------- #
# Retrieval
# --------------------------------------------------------------------------- #
class RetrievedChunk(BaseModel):
    """One chunk of one ticket or article, with enough metadata to cite it.

    Week-3 additions: source_file, article_id, product_area, last_updated
    are populated for article chunks; they are None for legacy ticket chunks.
    """

    chunk_id: str
    ticket_id: str          # article_id for article chunks
    subject: str            # article title for article chunks
    text: str
    score: float = Field(description="Cosine similarity, 0..1 — higher is closer.")

    # ticket-only fields
    category: Category | None = None
    priority: Priority | None = None
    status: Status | None = None
    product: str | None = None

    # article-only fields (Week-3 required metadata)
    source_file: str | None = None
    article_id: str | None = None
    product_area: str | None = None
    last_updated: str | None = None        # stored as ISO string in Chroma

    # Week-4 retrieval diagnostics (dense/BM25/RRF ranks)
    dense_rank: int | None = None
    bm25_rank: int | None = None
    dense_score: float | None = None
    bm25_score: float | None = None
    rrf_score: float | None = None
    cross_encoder_score: float | None = None
    cross_encoder_rank: int | None = None
    retrieval_mode: str | None = None


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2_000)
    top_k: int | None = Field(default=None, ge=1, le=50)
    mode: Literal["hybrid", "dense", "bm25", "mmr", "rerank"] = "hybrid"
    mmr_lambda: float = Field(default=0.7, ge=0.0, le=1.0)
    filters: TicketFilters | None = None
    article_filters: ArticleFilters | None = None


class SearchResponse(BaseModel):
    query: str
    mode: str = "hybrid"
    chunks: list[RetrievedChunk]


# --------------------------------------------------------------------------- #
# Chat / RAG
# --------------------------------------------------------------------------- #
class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8_000)
    history: list[ChatMessage] = Field(default_factory=list, max_length=40)
    filters: TicketFilters | None = None
    article_filters: ArticleFilters | None = None
    top_k: int | None = Field(default=None, ge=1, le=50)
    mode: Literal["hybrid", "dense", "bm25", "mmr", "rerank"] = "hybrid"
    image_base64: str | None = Field(default=None, description="Base64 encoded image string for multimodal analysis.")
    image_media_type: str | None = Field(default=None, description="Image MIME type (e.g. image/png, image/jpeg).")
    image_name: str | None = Field(default=None, description="Original filename of the attached image.")


class ChatResponse(BaseModel):
    answer: str
    sources: list[RetrievedChunk]
    model: str
    usage: dict[str, int] = Field(default_factory=dict)
    refused: bool = False


# --------------------------------------------------------------------------- #
# Triage (structured output)
# --------------------------------------------------------------------------- #
class TriageRequest(BaseModel):
    subject: str = Field(min_length=1, max_length=500)
    body: str = Field(min_length=1, max_length=8_000)


class TriageResult(BaseModel):
    """Schema handed to Claude via structured outputs — field docs steer the model."""

    category: Category
    priority: Priority
    sentiment: Sentiment
    summary: str = Field(description="One sentence, max 25 words.")
    suggested_tags: list[str] = Field(description="2-4 lowercase keyword tags.")
    reasoning: str = Field(description="Why this priority, referencing similar past tickets.")


class TriageResponse(BaseModel):
    result: TriageResult
    similar_tickets: list[RetrievedChunk]
    model: str


# --------------------------------------------------------------------------- #
# Reply drafting
# --------------------------------------------------------------------------- #
class DraftReplyRequest(BaseModel):
    subject: str = Field(min_length=1, max_length=500)
    body: str = Field(min_length=1, max_length=8_000)
    tone: Literal["friendly", "formal", "concise"] = "friendly"


class DraftReplyResponse(BaseModel):
    draft: str
    sources: list[RetrievedChunk]
    model: str


# --------------------------------------------------------------------------- #
# Ingestion
# --------------------------------------------------------------------------- #
class IngestRequest(BaseModel):
    reset: bool = Field(default=False, description="Drop the collection before indexing.")


class IngestStats(BaseModel):
    files_read: int
    tickets_indexed: int
    chunks_written: int
    duration_seconds: float
    collection: str


# --------------------------------------------------------------------------- #
# Stats / health
# --------------------------------------------------------------------------- #
class CollectionStats(BaseModel):
    collection: str
    chunk_count: int
    ticket_count: int
    by_category: dict[str, int]
    by_priority: dict[str, int]
    by_status: dict[str, int]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    vector_store: bool
    anthropic_configured: bool
    chunk_count: int

    # Which backend generation actually runs on right now, as opposed to what
    # LLM_PROVIDER asks for — the two differ whenever a provider is unreachable.
    llm_provider: Literal["anthropic", "ollama", "openai", "fallback"] = "fallback"
    ollama_available: bool = False
    model: str | None = None
