"""The RAG service: retrieve, build a prompt, call the LLM, return grounded output.

Three providers sit behind one interface, resolved per request by `_provider()`:
  anthropic  — Claude, best grounding and the only one with native structured output
  ollama     — a local model, free and offline, used for all four LLM paths
  fallback   — no model at all; deterministic templates over the retrieved chunks
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Iterator, Literal

import anthropic
from pydantic import ValidationError

from ..config import get_settings
from ..retrieval.retriever import Retriever, format_context
from ..telemetry.langfuse_client import log_rag_trace
from ..schemas import (
    Category,
    ChatMessage,
    ChatResponse,
    DraftReplyResponse,
    Priority,
    RetrievedChunk,
    Sentiment,
    TicketFilters,
    TriageResponse,
    TriageResult,
)
from . import prompts
from .claude_client import (
    CredentialsError,
    base_request_kwargs,
    extract_text,
    get_client,
    is_configured,
    refusal_message,
    was_refused,
)
from .ollama_client import (
    OllamaError,
    generate_ollama_chat,
    generate_ollama_json,
    is_ollama_available,
    stream_ollama_chat,
)
from .openai_client import (
    OpenAIClientError,
    generate_openai_chat,
    generate_openai_json,
    is_openai_available,
    stream_openai_chat,
)

logger = logging.getLogger(__name__)

HISTORY_TURNS = 8

Provider = Literal["anthropic", "ollama", "openai", "fallback"]

# Hand-written flat schema rather than `TriageResult.model_json_schema()`: pydantic
# emits the enums as `$ref`/`$defs`, and Ollama's grammar compiler is happiest with
# enums inlined. Values are pulled from the enums so the two cannot drift.
TRIAGE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "category": {"type": "string", "enum": [c.value for c in Category]},
        "priority": {"type": "string", "enum": [p.value for p in Priority]},
        "sentiment": {"type": "string", "enum": [s.value for s in Sentiment]},
        "summary": {"type": "string"},
        "suggested_tags": {"type": "array", "items": {"type": "string"}},
        "reasoning": {"type": "string"},
    },
    "required": [
        "category",
        "priority",
        "sentiment",
        "summary",
        "suggested_tags",
        "reasoning",
    ],
}

# Claude gets the schema through the API; a local model has to be told in words
# what the grammar is already forcing it to do, or it fills fields with filler.
TRIAGE_JSON_INSTRUCTION = (
    "\n\nRespond with a single JSON object and nothing else. Fields: category, "
    "priority, sentiment, summary, suggested_tags (2-4 lowercase keywords), reasoning."
)


def resolve_provider() -> Provider:
    """Resolve `LLM_PROVIDER` to a backend that can actually serve a request."""
    configured = get_settings().llm_provider
    if configured == "fallback":
        return "fallback"
    if configured == "anthropic" and is_configured():
        return "anthropic"
    if configured == "openai" and is_openai_available():
        return "openai"
    if configured == "ollama" and is_ollama_available():
        return "ollama"
    if is_ollama_available():
        return "ollama"
    return "fallback"


def active_model_name() -> str:
    """The model string /health should report for the resolved provider."""
    settings = get_settings()
    provider = resolve_provider()
    if provider == "anthropic":
        return settings.claude_model
    if provider == "openai":
        return f"openai/{settings.openai_model}"
    if provider == "ollama":
        return f"ollama/{settings.ollama_model}"
    return "local-extractor (100% free)"


class Answerer:
    """Wraps LLM calls that depend on retrieved ticket context."""

    def __init__(self, retriever: Retriever) -> None:
        self.retriever = retriever
        self.settings = get_settings()

    @property
    def client(self) -> anthropic.Anthropic:
        return get_client()

    def _provider(self) -> Provider:
        return resolve_provider()

    # ------------------------------------------------------------------ chat
    def answer(
        self,
        question: str,
        history: list[ChatMessage] | None = None,
        filters: TicketFilters | None = None,
        article_filters: Any | None = None,
        top_k: int | None = None,
        mode: str | None = None,
        image_base64: str | None = None,
        image_media_type: str | None = None,
        image_name: str | None = None,
    ) -> ChatResponse:
        """Answer a question over the ticket archive, with citations and optional multimodal image."""
        search_query = question
        final_question = question
        if image_name:
            err_match = re.search(r"(?:err[-_]?)(\d{4})", image_name, re.IGNORECASE)
            if err_match and not re.search(r"ERR-\d{4}", question, re.IGNORECASE):
                err_code = f"ERR-{err_match.group(1)}"
                search_query = f"{question} {err_code}"
                final_question = f"{question} {err_code} (from uploaded screenshot)"

        chunks = self.retriever.retrieve(
            search_query, top_k=top_k, filters=filters, article_filters=article_filters, mode=mode
        )
        provider = self._provider()

        t_start = time.perf_counter()
        chat_response: ChatResponse

        if provider == "anthropic":
            messages = self._chat_messages(
                final_question, chunks, history, image_base64=image_base64, image_media_type=image_media_type
            )
            response = self.client.beta.messages.create(
                system=_cached_system(prompts.CHAT_SYSTEM),
                messages=messages,
                **base_request_kwargs(),
            )

            if was_refused(response):
                chat_response = ChatResponse(
                    answer=refusal_message(response),
                    sources=[],
                    model=getattr(response, "model", self.settings.claude_model),
                    refused=True,
                )
            else:
                usage = getattr(response, "usage", None)
                chat_response = ChatResponse(
                    answer=extract_text(response),
                    sources=chunks,
                    model=getattr(response, "model", self.settings.claude_model),
                    usage={
                        "input_tokens": getattr(usage, "input_tokens", 0) or 0,
                        "output_tokens": getattr(usage, "output_tokens", 0) or 0,
                        "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", 0) or 0,
                    },
                )

        elif provider == "openai":
            try:
                answer_text = generate_openai_chat(
                    self._chat_messages(
                        final_question, chunks, history, image_base64=image_base64, image_media_type=image_media_type
                    ),
                    prompts.CHAT_SYSTEM,
                )
                chat_response = ChatResponse(
                    answer=answer_text,
                    sources=chunks,
                    model=f"openai/{self.settings.openai_model}",
                    usage=_no_usage(),
                )
            except OpenAIClientError:
                logger.exception("OpenAI/Groq chat failed — falling back to the local extractor")
                chat_response = ChatResponse(
                    answer=_build_fallback_answer(final_question, chunks),
                    sources=chunks,
                    model="local-extractor (100% free)",
                    usage=_no_usage(),
                )

        elif provider == "ollama":
            try:
                answer_text = generate_ollama_chat(
                    self._chat_messages(
                        final_question, chunks, history, image_base64=image_base64, image_media_type=image_media_type
                    ),
                    prompts.CHAT_SYSTEM,
                )
                chat_response = ChatResponse(
                    answer=answer_text,
                    sources=chunks,
                    model=f"ollama/{self.settings.ollama_model}",
                    usage=_no_usage(),
                )
            except OllamaError:
                logger.exception("Ollama chat failed — falling back to the local extractor")
                chat_response = ChatResponse(
                    answer=_build_fallback_answer(final_question, chunks),
                    sources=chunks,
                    model="local-extractor (100% free)",
                    usage=_no_usage(),
                )

        else:
            # Free local RAG extractor: no server, no key, no tokens.
            chat_response = ChatResponse(
                answer=_build_fallback_answer(final_question, chunks),
                sources=chunks,
                model="local-extractor (100% free)",
                usage=_no_usage(),
            )

        elapsed_ms = (time.perf_counter() - t_start) * 1000
        # Telemetry: push trace to Langfuse if configured
        log_rag_trace(
            query=final_question,
            retrieved_chunks=chunks,
            answer=chat_response.answer,
            model=chat_response.model,
            latency_ms=round(elapsed_ms, 2),
            metadata={"refused": chat_response.refused, "provider": provider},
        )

        return chat_response

    def answer_stream(
        self,
        question: str,
        history: list[ChatMessage] | None = None,
        filters: TicketFilters | None = None,
        article_filters: Any | None = None,
        top_k: int | None = None,
        mode: str | None = None,
        image_base64: str | None = None,
        image_media_type: str | None = None,
        image_name: str | None = None,
    ) -> Iterator[str]:
        """Server-sent-event generator for the chat UI."""
        search_query = question
        final_question = question
        if image_name:
            err_match = re.search(r"(?:err[-_]?)(\d{4})", image_name, re.IGNORECASE)
            if err_match and not re.search(r"ERR-\d{4}", question, re.IGNORECASE):
                err_code = f"ERR-{err_match.group(1)}"
                search_query = f"{question} {err_code}"
                final_question = f"{question} {err_code} (from uploaded screenshot)"

        chunks = self.retriever.retrieve(
            search_query, top_k=top_k, filters=filters, article_filters=article_filters, mode=mode
        )
        yield _sse("sources", [c.model_dump(mode="json") for c in chunks])
        provider = self._provider()

        if provider == "anthropic":
            messages = self._chat_messages(
                final_question, chunks, history, image_base64=image_base64, image_media_type=image_media_type
            )
            try:
                with self.client.beta.messages.stream(
                    system=_cached_system(prompts.CHAT_SYSTEM),
                    messages=messages,
                    **base_request_kwargs(stream=True),
                ) as stream:
                    for text in stream.text_stream:
                        yield _sse("delta", {"text": text})

                    final = stream.get_final_message()

                if was_refused(final):
                    yield _sse("error", {"message": refusal_message(final), "refused": True})
                else:
                    usage = getattr(final, "usage", None)
                    yield _sse(
                        "done",
                        {
                            "model": getattr(final, "model", self.settings.claude_model),
                            "output_tokens": getattr(usage, "output_tokens", 0) or 0,
                        },
                    )
            except CredentialsError as exc:
                yield _sse("error", {"message": str(exc)})
            except anthropic.APIError as exc:
                logger.exception("stream failed")
                yield _sse("error", {"message": f"Generation failed: {exc}"})
            return

        if provider == "openai":
            for token in stream_openai_chat(
                self._chat_messages(
                    question, chunks, history, image_base64=image_base64, image_media_type=image_media_type
                ),
                prompts.CHAT_SYSTEM,
            ):
                yield _sse("delta", {"text": token})
            yield _sse(
                "done",
                {"model": f"openai/{self.settings.openai_model}", "output_tokens": 0},
            )
            return

        if provider == "ollama":
            for token in stream_ollama_chat(
                self._chat_messages(
                    question, chunks, history, image_base64=image_base64, image_media_type=image_media_type
                ),
                prompts.CHAT_SYSTEM,
            ):
                yield _sse("delta", {"text": token})
            yield _sse(
                "done",
                {"model": f"ollama/{self.settings.ollama_model}", "output_tokens": 0},
            )
            return

        # Local RAG extractor stream (free, zero-token).
        yield _sse("delta", {"text": _build_fallback_answer(question, chunks)})
        yield _sse("done", {"model": "local-extractor (100% free)", "output_tokens": 0})

    def _chat_messages(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        history: list[ChatMessage] | None,
        image_base64: str | None = None,
        image_media_type: str | None = None,
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [
            {"role": message.role, "content": message.content}
            for message in (history or [])[-HISTORY_TURNS:]
        ]
        user_text = prompts.CHAT_USER_TEMPLATE.format(
            context=format_context(chunks), question=question
        )
        if image_base64 and self._provider() == "anthropic":
            content_blocks = [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": image_media_type or "image/png",
                        "data": image_base64,
                    },
                },
                {"type": "text", "text": user_text},
            ]
            messages.append({"role": "user", "content": content_blocks})
        elif image_base64 and (self._provider() in {"ollama", "openai"}):
            messages.append({"role": "user", "content": user_text, "images": [image_base64]})
        else:
            messages.append({"role": "user", "content": user_text})
        return messages

    # ---------------------------------------------------------------- triage
    def triage(self, subject: str, body: str) -> TriageResponse:
        similar = self.retriever.similar_to_ticket(subject, body, top_k=4)
        provider = self._provider()
        user_message = prompts.TRIAGE_USER_TEMPLATE.format(
            context=format_context(similar), subject=subject, body=body
        )

        if provider == "anthropic":
            response = self.client.messages.parse(
                model=self.settings.claude_model,
                max_tokens=self.settings.claude_max_tokens,
                thinking={"type": "adaptive"},
                output_config={"effort": self.settings.claude_effort},
                system=_cached_system(prompts.TRIAGE_SYSTEM),
                messages=[{"role": "user", "content": user_message}],
                output_format=TriageResult,
            )

            result = response.parsed_output
            if result is None:
                raise ValueError("Triage returned no structured result")

            return TriageResponse(
                result=result,
                similar_tickets=similar,
                model=getattr(response, "model", self.settings.claude_model),
            )

        if provider == "openai":
            try:
                raw = generate_openai_json(
                    [{"role": "user", "content": user_message}],
                    prompts.TRIAGE_SYSTEM + TRIAGE_JSON_INSTRUCTION,
                    TRIAGE_JSON_SCHEMA,
                )
                result = TriageResult.model_validate(raw)
            except (OpenAIClientError, ValidationError):
                logger.exception("OpenAI triage failed — falling back to local heuristic")
            else:
                return TriageResponse(
                    result=result,
                    similar_tickets=similar,
                    model=f"openai/{self.settings.openai_model}",
                )

        if provider == "ollama":
            try:
                raw = generate_ollama_json(
                    [{"role": "user", "content": user_message}],
                    prompts.TRIAGE_SYSTEM + TRIAGE_JSON_INSTRUCTION,
                    TRIAGE_JSON_SCHEMA,
                )
                result = TriageResult.model_validate(raw)
            except (OllamaError, ValidationError):
                logger.exception("Ollama triage failed — falling back to the local heuristic")
            else:
                return TriageResponse(
                    result=result,
                    similar_tickets=similar,
                    model=f"ollama/{self.settings.ollama_model}",
                )

        # Free local triage estimated from the retrieved similar tickets.
        return TriageResponse(
            result=_build_local_triage(subject, body, similar),
            similar_tickets=similar,
            model="local-triage (100% free)",
        )

    # ----------------------------------------------------------------- draft
    def draft_reply(self, subject: str, body: str, tone: str = "friendly") -> DraftReplyResponse:
        similar = self.retriever.similar_to_ticket(subject, body, top_k=4)
        tone_instruction = prompts.TONE_INSTRUCTIONS.get(
            tone, prompts.TONE_INSTRUCTIONS["friendly"]
        )
        provider = self._provider()
        system_prompt = prompts.DRAFT_SYSTEM.format(tone_instruction=tone_instruction)
        user_message = prompts.DRAFT_USER_TEMPLATE.format(
            context=format_context(similar), subject=subject, body=body
        )

        if provider == "anthropic":
            response = self.client.beta.messages.create(
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
                **base_request_kwargs(),
            )

            if was_refused(response):
                raise ValueError(refusal_message(response))

            return DraftReplyResponse(
                draft=extract_text(response),
                sources=similar,
                model=getattr(response, "model", self.settings.claude_model),
            )

        if provider == "openai":
            try:
                draft_text = generate_openai_chat(
                    [{"role": "user", "content": user_message}], system_prompt
                )
            except OpenAIClientError:
                logger.exception("OpenAI draft failed — falling back to local drafter")
            else:
                return DraftReplyResponse(
                    draft=draft_text,
                    sources=similar,
                    model=f"openai/{self.settings.openai_model}",
                )

        if provider == "ollama":
            try:
                draft_text = generate_ollama_chat(
                    [{"role": "user", "content": user_message}], system_prompt
                )
            except OllamaError:
                logger.exception("Ollama draft failed — falling back to the local drafter")
            else:
                return DraftReplyResponse(
                    draft=draft_text,
                    sources=similar,
                    model=f"ollama/{self.settings.ollama_model}",
                )

        # Free local reply drafting from similar resolved tickets.
        return DraftReplyResponse(
            draft=_build_local_draft(subject, body, similar, tone),
            sources=similar,
            model="local-drafter (100% free)",
        )


def _build_fallback_answer(question: str, chunks: list[RetrievedChunk]) -> str:
    """A structured, grounded answer assembled from retrieved chunks — no model."""
    max_dense = max((c.dense_score or c.score or 0.0) for c in chunks) if chunks else 0.0

    if not chunks or max_dense < 0.25:
        return (
            f"❌ **No matching support documentation found** for: *\"{question}\"*\n\n"
            "The indexed knowledge base is focused on billing migration and customer support topics, including:\n"
            "• **Error Codes**: `ERR-4032`, `ERR-4001`, `ERR-4010`, `ERR-4040`, `ERR-4030`\n"
            "• **Invoices & Credits**: Cutover rules, credit expiration, and adjustments\n"
            "• **Webhooks & APIs**: Header replacements (`Authorization: Bearer`), secret rotation\n"
            "• **SSO & Permissions**: SAML attribute configuration and user role mapping\n\n"
            "💡 *Tip: To enable general conversational Q&A on any topic, start local Ollama (`ollama run llama3.2`) or add an Anthropic API key in `.env`.*"
        )

    lines = [f"Based on **{len(chunks)}** relevant support article(s) in your index:\n"]
    for idx, chunk in enumerate(chunks, 1):
        title = chunk.subject or chunk.ticket_id
        lines.append(f"**[{idx}] {title}** (`{chunk.chunk_id}`)")
        snippet = chunk.text.replace("\n", " ").strip()
        lines.append(f"> {snippet}\n")

    lines.append("---")
    lines.append(
        "*💡 Note: Running in 100% free mode. To enable conversational AI locally, "
        "start Ollama via `ollama run llama3.2`.*"
    )
    return "\n".join(lines)


def _build_local_triage(subject: str, body: str, similar: list[RetrievedChunk]) -> TriageResult:
    category = "technical"
    priority = "medium"
    if similar:
        if similar[0].category:
            category = similar[0].category
        if similar[0].priority:
            priority = similar[0].priority

    return TriageResult(
        category=category,
        priority=priority,
        sentiment="neutral",
        summary=f"Incoming ticket: '{subject}' matched {len(similar)} similar historical tickets.",
        suggested_tags=[category, priority, "free-mode"],
        reasoning=(
            f"Matched against past ticket "
            f"{similar[0].ticket_id if similar else 'archive'} based on vector similarity."
        ),
    )


def _build_local_draft(subject: str, body: str, similar: list[RetrievedChunk], tone: str) -> str:
    greeting = "Hello," if tone == "formal" else "Hi there,"
    if not similar:
        return (
            f"{greeting}\n\nThank you for reaching out regarding '{subject}'. We have "
            f"received your request and our team is investigating.\n\n"
            f"Best regards,\nSupport Team"
        )

    best = similar[0]
    return (
        f"{greeting}\n\n"
        f"Thank you for contacting support regarding '{subject}'.\n\n"
        f"Based on similar resolved issues (such as ticket {best.ticket_id} - '{best.subject}'), "
        f"here is the recommended solution:\n\n"
        f"{best.text}\n\n"
        f"Please let us know if this helps resolve your issue!\n\n"
        f"Best regards,\nCustomer Support"
    )


def _no_usage() -> dict[str, int]:
    return {"input_tokens": 0, "output_tokens": 0, "cache_read_input_tokens": 0}


def _cached_system(text: str) -> list[dict[str, Any]]:
    return [{"type": "text", "text": text, "cache_control": {"type": "ephemeral"}}]


def _sse(event: str, data: Any) -> str:
    return f"event: {event}\ndata: {json.dumps(data, default=str)}\n\n"
