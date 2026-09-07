"""Telemetry and observability module supporting Langfuse and local tracing."""

from .langfuse_client import get_langfuse_client, is_langfuse_configured, log_rag_trace

__all__ = ["get_langfuse_client", "is_langfuse_configured", "log_rag_trace"]
