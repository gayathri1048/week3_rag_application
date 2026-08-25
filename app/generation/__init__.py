"""Generation: the Claude client, prompt templates, and the RAG answering service."""

from .answerer import Answerer
from .claude_client import get_client, is_configured

__all__ = ["Answerer", "get_client", "is_configured"]
