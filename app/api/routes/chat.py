"""RAG endpoints: chat (blocking + streaming), triage, reply drafting, raw search."""

from __future__ import annotations

import logging

import anthropic
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse

from ...dependencies import get_answerer, get_retriever
from ...generation.answerer import Answerer
from ...generation.claude_client import CredentialsError
from ...retrieval.retriever import Retriever
from ...schemas import (
    ChatRequest,
    ChatResponse,
    DraftReplyRequest,
    DraftReplyResponse,
    SearchRequest,
    SearchResponse,
    TriageRequest,
    TriageResponse,
)

logger = logging.getLogger(__name__)
router = APIRouter()

# Proxies buffer SSE by default, which makes a streamed answer arrive all at once.
SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _api_error(exc: anthropic.APIError) -> HTTPException:
    """Map an SDK error onto a status code the frontend can act on."""
    if isinstance(exc, anthropic.AuthenticationError):
        return HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Anthropic credentials missing or invalid — check ANTHROPIC_API_KEY.",
        )
    if isinstance(exc, anthropic.RateLimitError):
        return HTTPException(status.HTTP_429_TOO_MANY_REQUESTS, "Rate limited — retry shortly.")
    if isinstance(exc, anthropic.APIConnectionError):
        return HTTPException(status.HTTP_504_GATEWAY_TIMEOUT, "Could not reach the Anthropic API.")
    return HTTPException(status.HTTP_502_BAD_GATEWAY, f"Generation failed: {exc}")


@router.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    answerer: Answerer = Depends(get_answerer),
) -> ChatResponse:
    """Answer a question about the ticket archive. Returns the full answer at once."""
    try:
        return answerer.answer(
            question=request.message,
            history=request.history,
            filters=request.filters,
            top_k=request.top_k,
        )
    except CredentialsError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    except anthropic.APIError as exc:
        logger.exception("chat failed")
        raise _api_error(exc) from exc


@router.post("/chat/stream")
def chat_stream(
    request: ChatRequest,
    answerer: Answerer = Depends(get_answerer),
) -> StreamingResponse:
    """Same as /chat but streamed as SSE.

    Events, in order: `sources` (once), `delta` (many), then `done` or `error`.
    """
    return StreamingResponse(
        answerer.answer_stream(
            question=request.message,
            history=request.history,
            filters=request.filters,
            top_k=request.top_k,
        ),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )


@router.post("/triage", response_model=TriageResponse)
def triage(
    request: TriageRequest,
    answerer: Answerer = Depends(get_answerer),
) -> TriageResponse:
    """Classify an incoming ticket against similar resolved history."""
    try:
        return answerer.triage(subject=request.subject, body=request.body)
    except CredentialsError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    except anthropic.APIError as exc:
        logger.exception("triage failed")
        raise _api_error(exc) from exc


@router.post("/draft-reply", response_model=DraftReplyResponse)
def draft_reply(
    request: DraftReplyRequest,
    answerer: Answerer = Depends(get_answerer),
) -> DraftReplyResponse:
    """Draft an agent reply grounded in how similar tickets were resolved."""
    try:
        return answerer.draft_reply(
            subject=request.subject, body=request.body, tone=request.tone
        )
    except CredentialsError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    except anthropic.APIError as exc:
        logger.exception("draft failed")
        raise _api_error(exc) from exc


@router.post("/search", response_model=SearchResponse)
def search(
    request: SearchRequest,
    retriever: Retriever = Depends(get_retriever),
) -> SearchResponse:
    """Retrieval only — no Claude call. Useful for debugging what the model sees."""
    chunks = retriever.retrieve(
        query=request.query, top_k=request.top_k, filters=request.filters
    )
    return SearchResponse(query=request.query, chunks=chunks)
