"""Index management: rebuild from documents/, upload new tickets, delete one."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from ...config import get_settings
from ...dependencies import get_vector_store
from ...ingestion.pipeline import ingest as run_ingest
from ...retrieval.vector_store import VectorStore
from ...schemas import IngestRequest, IngestStats

logger = logging.getLogger(__name__)
router = APIRouter()

ALLOWED_UPLOAD_SUFFIXES = {".json", ".jsonl", ".csv", ".md"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]")


@router.post("/ingest", response_model=IngestStats)
def ingest_documents(
    request: IngestRequest,
    store: VectorStore = Depends(get_vector_store),
) -> IngestStats:
    """(Re)index everything in `documents/`.

    Idempotent — chunk ids are derived from ticket ids, so an unchanged ticket is
    overwritten with identical content. Pass `reset` to drop the collection first,
    which is needed when a ticket shrinks to fewer chunks than it had before.
    """
    try:
        return run_ingest(store, reset=request.reset)
    except FileNotFoundError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    except Exception as exc:
        logger.exception("ingest failed")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"Ingest failed: {exc}") from exc


@router.post("/ingest/upload", response_model=IngestStats)
async def upload_documents(
    files: list[UploadFile] = File(...),
    store: VectorStore = Depends(get_vector_store),
) -> IngestStats:
    """Save uploaded ticket files into `documents/` and reindex.

    Filenames are sanitised and forced into `documents/` — an upload can't write
    outside that directory.
    """
    settings = get_settings()
    documents_dir = settings.documents_dir
    documents_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    for upload in files:
        name = SAFE_NAME.sub("_", Path(upload.filename or "upload").name)
        suffix = Path(name).suffix.lower()

        if suffix not in ALLOWED_UPLOAD_SUFFIXES:
            raise HTTPException(
                status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                f"{name}: expected one of {sorted(ALLOWED_UPLOAD_SUFFIXES)}",
            )

        content = await upload.read()
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                f"{name} exceeds the {MAX_UPLOAD_BYTES // 1024 // 1024}MB limit.",
            )
        if suffix == ".json":
            try:
                json.loads(content.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise HTTPException(
                    status.HTTP_400_BAD_REQUEST, f"{name} is not valid JSON: {exc}"
                ) from exc

        destination = documents_dir / name
        destination.write_bytes(content)
        saved.append(destination)
        logger.info("saved upload %s (%d bytes)", destination.name, len(content))

    if not saved:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "No files were uploaded.")

    return run_ingest(store, reset=False)


@router.delete("/tickets/{ticket_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_ticket(ticket_id: str, store: VectorStore = Depends(get_vector_store)) -> None:
    """Remove a ticket's chunks from the index.

    The source file in `documents/` is untouched, so a later reindex brings it back.
    """
    store.delete_ticket(ticket_id)
    logger.info("deleted chunks for ticket %s", ticket_id)
