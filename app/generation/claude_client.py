"""Anthropic client construction and shared request settings."""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

import anthropic

from ..config import get_settings

logger = logging.getLogger(__name__)

# `fallbacks: "default"` re-runs a policy-declined request on Anthropic's recommended
# substitute model server-side, routed by refusal category. This header gates the
# scalar form; the array form (`[{"model": ...}]`) uses -2026-06-01 instead.
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class CredentialsError(RuntimeError):
    """No Anthropic credential could be resolved.

    The SDK raises a bare TypeError for this, which is indistinguishable from a
    programming bug at the call site — this makes it catchable and mappable to a 503.
    """



NO_CREDENTIALS_MESSAGE = (
    "No Anthropic credentials found. Set ANTHROPIC_API_KEY in .env or run "
    "`ant auth login` — or set LLM_PROVIDER=ollama to generate locally for free. "
    "Retrieval and search work without a credential; chat, triage, and reply "
    "drafting do not."
)


@lru_cache
def get_client() -> anthropic.Anthropic:
    """Return a cached Anthropic client.

    With no `ANTHROPIC_API_KEY` the SDK still resolves credentials from
    ANTHROPIC_AUTH_TOKEN or an `ant auth login` profile, so a bare constructor is
    the right default.

    The credential check happens here rather than being left to the SDK: the
    constructor succeeds regardless, and the SDK only raises its bare TypeError
    from inside `_validate_headers` at request time — too deep for the route
    handlers to map onto a 503, so it would surface as an unhandled 500.
    """
    settings = get_settings()
    if not settings.anthropic_api_key and not is_configured():
        raise CredentialsError(NO_CREDENTIALS_MESSAGE)
    try:
        if settings.anthropic_api_key:
            return anthropic.Anthropic(api_key=settings.anthropic_api_key)
        return anthropic.Anthropic()
    except TypeError as exc:  # belt and braces if the SDK ever moves the check
        raise CredentialsError(NO_CREDENTIALS_MESSAGE) from exc


def is_configured() -> bool:
    """Whether any Anthropic credential is likely available — used by /health."""
    settings = get_settings()
    if settings.anthropic_api_key or os.getenv("ANTHROPIC_AUTH_TOKEN"):
        return True
    # An `ant auth login` profile lives on disk and needs no env var.
    return (Path.home() / ".config" / "anthropic").exists()


def base_request_kwargs(stream: bool = False) -> dict[str, object]:
    """Request parameters shared by every call site.

    - `thinking: adaptive` is the only on-mode on current models; Claude decides how
      much to think per request. Effort is what you tune, not a token budget.
    - `max_tokens` covers thinking *and* visible text together, so it is generous;
      the streaming path gets more room since HTTP timeouts aren't a concern there.
    - Fallbacks are opt-in — without them a declined request simply stops.
    """
    settings = get_settings()
    return {
        "model": settings.claude_model,
        "max_tokens": 64_000 if stream else settings.claude_max_tokens,
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": settings.claude_effort},
        "betas": [FALLBACK_BETA],
        "fallbacks": "default",
    }


def extract_text(response: object) -> str:
    """Concatenate the text blocks of a response, ignoring thinking blocks."""
    parts = [
        block.text
        for block in getattr(response, "content", [])
        if getattr(block, "type", None) == "text"
    ]
    return "".join(parts).strip()


def was_refused(response: object) -> bool:
    """True when the whole fallback chain declined the request."""
    return getattr(response, "stop_reason", None) == "refusal"


def refusal_message(response: object) -> str:
    """A user-facing sentence for a refused request."""
    details = getattr(response, "stop_details", None)
    category = getattr(details, "category", None) if details else None
    logger.warning("request refused (category=%s)", category)
    return (
        "I can't help with that request. If you were asking about the support ticket "
        "history, try rephrasing the question."
    )
