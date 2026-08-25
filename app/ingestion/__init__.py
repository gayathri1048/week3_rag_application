"""Ingestion: read raw tickets from disk, chunk them, write them to the vector store."""

from .chunker import chunk_ticket
from .loaders import load_tickets
from .pipeline import ingest

__all__ = ["chunk_ticket", "load_tickets", "ingest"]
