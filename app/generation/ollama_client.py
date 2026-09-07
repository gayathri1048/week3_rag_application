"""Ollama client for local inference — no API key, no token cost, no network.

Mirrors the surface `claude_client` exposes: an availability probe for /health,
a blocking call, a streaming call, and a JSON-schema-constrained call for the
structured triage output. Failures raise `OllamaError` so the caller can decide
between degrading to the template path and surfacing a 502.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Iterator

import httpx

from ..config import get_settings

logger = logging.getLogger(__name__)

# Generation is slower locally than over the API — a 7B model on CPU can take
# well over the httpx default of 5s just to emit its first token.
REQUEST_TIMEOUT = 120.0
PROBE_TIMEOUT = 2.0


class OllamaError(RuntimeError):
    """The local Ollama server was unreachable, errored, or returned junk."""


def is_ollama_available() -> bool:
    """Whether the local Ollama server is up — used by /health and provider routing."""
    settings = get_settings()
    try:
        response = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=PROBE_TIMEOUT)
        return response.status_code == 200
    except Exception:
        return False


def get_available_models() -> list[str]:
    """The models currently pulled into the local Ollama instance."""
    settings = get_settings()
    try:
        response = httpx.get(f"{settings.ollama_base_url}/api/tags", timeout=3.0)
        if response.status_code == 200:
            return [m.get("name", "") for m in response.json().get("models", [])]
    except Exception:
        logger.debug("could not list Ollama models", exc_info=True)
    return []


def _payload(
    messages: list[dict[str, Any]],
    system_prompt: str,
    *,
    stream: bool,
    fmt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    settings = get_settings()
    payload: dict[str, Any] = {
        "model": settings.ollama_model,
        "messages": [{"role": "system", "content": system_prompt}] + messages,
        "stream": stream,
    }
    if fmt is not None:
        # Ollama constrains decoding to this JSON Schema (>= 0.5), which is the
        # local equivalent of `messages.parse` — the model cannot emit invalid JSON.
        payload["format"] = fmt
        payload["options"] = {"temperature": 0}
    return payload


def stream_ollama_chat(
    messages: list[dict[str, Any]],
    system_prompt: str,
) -> Iterator[str]:
    """Stream chat tokens from the local Ollama instance.

    Yields an inline error string rather than raising: the caller is already
    mid-SSE-response, so there is no status code left to change.
    """
    settings = get_settings()
    url = f"{settings.ollama_base_url}/api/chat"

    try:
        with httpx.stream(
            "POST", url, json=_payload(messages, system_prompt, stream=True), timeout=REQUEST_TIMEOUT
        ) as response:
            if response.status_code != 200:
                logger.error("Ollama stream returned %s", response.status_code)
                yield f"[Ollama error {response.status_code} — is `{settings.ollama_model}` pulled?]"
                return

            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    delta = json.loads(line).get("message", {}).get("content", "")
                except json.JSONDecodeError:
                    continue
                if delta:
                    yield delta
    except Exception as exc:
        logger.exception("failed streaming from Ollama")
        yield (
            f"[Ollama connection error: {exc}. Is Ollama running? "
            f"Start it with `ollama run {settings.ollama_model}`.]"
        )


def generate_ollama_chat(
    messages: list[dict[str, Any]],
    system_prompt: str,
) -> str:
    """Blocking chat completion from the local Ollama instance."""
    text = _post(_payload(messages, system_prompt, stream=False))
    if not text:
        raise OllamaError("Ollama returned an empty response.")
    return text


def generate_ollama_json(
    messages: list[dict[str, Any]],
    system_prompt: str,
    json_schema: dict[str, Any],
) -> dict[str, Any]:
    """Schema-constrained completion, returned as a parsed dict.

    Small local models still occasionally wrap the object in prose despite the
    grammar, so the text is parsed defensively rather than trusted outright.
    """
    text = _post(_payload(messages, system_prompt, stream=False, fmt=json_schema))
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise OllamaError(f"Ollama did not return valid JSON: {text[:200]!r}") from exc
    if not isinstance(parsed, dict):
        raise OllamaError(f"Expected a JSON object from Ollama, got {type(parsed).__name__}.")
    return parsed


def _post(payload: dict[str, Any]) -> str:
    """POST /api/chat and return the assistant message content."""
    settings = get_settings()
    url = f"{settings.ollama_base_url}/api/chat"
    try:
        response = httpx.post(url, json=payload, timeout=REQUEST_TIMEOUT)
    except httpx.HTTPError as exc:
        raise OllamaError(
            f"Could not reach Ollama at {settings.ollama_base_url} ({exc}). "
            f"Start it with `ollama run {settings.ollama_model}`."
        ) from exc

    if response.status_code != 200:
        raise OllamaError(
            f"Ollama returned {response.status_code}: {response.text[:200]} "
            f"(is model `{settings.ollama_model}` pulled?)"
        )

    try:
        return response.json().get("message", {}).get("content", "").strip()
    except json.JSONDecodeError as exc:
        raise OllamaError("Ollama returned a malformed response body.") from exc
