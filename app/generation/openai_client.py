"""OpenAI-compatible client for cloud endpoints (Groq, OpenRouter, OpenAI, vLLM, etc.).

Allows zero-local-download inference with ultra-fast cloud vision and text models.
Supports streaming SSE, JSON schema constraints, and multimodal base64 image inspection.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Iterator

import httpx

from ..config import get_settings

logger = logging.getLogger(__name__)

REQUEST_TIMEOUT = 60.0
PROBE_TIMEOUT = 3.0


class OpenAIClientError(RuntimeError):
    """The remote OpenAI-compatible endpoint returned an error or was unreachable."""


def is_openai_available() -> bool:
    """Check if the remote OpenAI/Groq endpoint is configured with a key or reachable."""
    settings = get_settings()
    if not settings.openai_api_key:
        return False
    return True


def _headers() -> dict[str, str]:
    settings = get_settings()
    headers = {"Content-Type": "application/json"}
    if settings.openai_api_key:
        headers["Authorization"] = f"Bearer {settings.openai_api_key}"
    return headers


def _build_openai_messages(
    messages: list[dict[str, Any]],
    system_prompt: str,
    include_images: bool = True,
) -> list[dict[str, Any]]:
    """Format messages for OpenAI/Groq API."""
    formatted: list[dict[str, Any]] = [{"role": "system", "content": system_prompt}]
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        images = m.get("images", [])

        if images and include_images:
            content_blocks: list[dict[str, Any]] = [{"type": "text", "text": str(content)}]
            for img_b64 in images:
                content_blocks.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{img_b64}" if not img_b64.startswith("data:") else img_b64
                    }
                })
            formatted.append({"role": role, "content": content_blocks})
        else:
            formatted.append({"role": role, "content": str(content)})

    return formatted


def generate_openai_chat(
    messages: list[dict[str, Any]],
    system_prompt: str,
) -> str:
    """Blocking chat completion from remote OpenAI/Groq endpoint with auto-fallback for text models."""
    settings = get_settings()
    url = f"{settings.openai_base_url.rstrip('/')}/chat/completions"

    # Try first with image blocks if present
    for try_images in [True, False]:
        payload = {
            "model": settings.openai_model,
            "messages": _build_openai_messages(messages, system_prompt, include_images=try_images),
            "temperature": 0.2,
        }

        try:
            response = httpx.post(url, headers=_headers(), json=payload, timeout=REQUEST_TIMEOUT)
        except httpx.HTTPError as exc:
            raise OpenAIClientError(f"Could not reach {settings.openai_base_url}: {exc}") from exc

        if response.status_code == 200:
            try:
                data = response.json()
                return data["choices"][0]["message"]["content"].strip()
            except Exception as exc:
                raise OpenAIClientError("Malformed JSON response from OpenAI-compatible endpoint.") from exc

        # If Groq text model rejected image block ("content must be a string"), retry as string
        if response.status_code == 400 and try_images and "content must be a string" in response.text:
            continue

        raise OpenAIClientError(f"OpenAI/Groq API error ({response.status_code}): {response.text[:300]}")

    raise OpenAIClientError("OpenAI/Groq completion failed.")


def stream_openai_chat(
    messages: list[dict[str, Any]],
    system_prompt: str,
) -> Iterator[str]:
    """Streaming chat completion yielding text deltas."""
    settings = get_settings()
    url = f"{settings.openai_base_url.rstrip('/')}/chat/completions"

    # For streaming, format as string content for maximum compatibility
    payload = {
        "model": settings.openai_model,
        "messages": _build_openai_messages(messages, system_prompt, include_images=False),
        "temperature": 0.2,
        "stream": True,
    }

    try:
        with httpx.stream("POST", url, headers=_headers(), json=payload, timeout=REQUEST_TIMEOUT) as response:
            if response.status_code != 200:
                err_body = response.read().decode(errors="replace")
                logger.error("OpenAI/Groq stream returned %s: %s", response.status_code, err_body[:200])
                yield f"[API Error {response.status_code}: {err_body[:150]}]"
                return

            for line in response.iter_lines():
                if not line:
                    continue
                if line.startswith("data: "):
                    line = line[6:].strip()
                if line == "[DONE]":
                    break
                try:
                    chunk = json.loads(line)
                    delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    if delta:
                        yield delta
                except json.JSONDecodeError:
                    continue
    except Exception as exc:
        logger.exception("failed streaming from OpenAI/Groq endpoint")
        yield f"[Connection error: {exc}]"


def generate_openai_json(
    messages: list[dict[str, Any]],
    system_prompt: str,
    json_schema: dict[str, Any],
) -> dict[str, Any]:
    """Schema-constrained completion via response_format or json_object instruction."""
    settings = get_settings()
    url = f"{settings.openai_base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": settings.openai_model,
        "messages": _build_openai_messages(messages, system_prompt + "\n\nRespond ONLY with a valid JSON object matching the required schema."),
        "response_format": {"type": "json_object"},
        "temperature": 0.0,
    }

    try:
        response = httpx.post(url, headers=_headers(), json=payload, timeout=REQUEST_TIMEOUT)
    except httpx.HTTPError as exc:
        raise OpenAIClientError(f"Could not reach {settings.openai_base_url}: {exc}") from exc

    if response.status_code != 200:
        raise OpenAIClientError(f"OpenAI/Groq API error ({response.status_code}): {response.text[:300]}")

    try:
        raw_text = response.json()["choices"][0]["message"]["content"].strip()
        return json.loads(raw_text)
    except Exception as exc:
        raise OpenAIClientError(f"Failed to parse JSON from response: {exc}") from exc
