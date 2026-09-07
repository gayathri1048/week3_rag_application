"""Index management: rebuild from documents/, upload new tickets/articles, delete one."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from ...config import get_settings
from ...dependencies import get_retriever, get_vector_store
from ...ingestion.article_pipeline import ingest_articles
from ...ingestion.pipeline import ingest as run_ingest
from ...retrieval.retriever import Retriever
from ...retrieval.vector_store import VectorStore
from ...schemas import IngestRequest, IngestStats

logger = logging.getLogger(__name__)
router = APIRouter()

ALLOWED_UPLOAD_SUFFIXES = {".json", ".jsonl", ".csv", ".md"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]")


def _run_appropriate_ingest(store: VectorStore, reset: bool = False) -> IngestStats:
    """Select article ingestion or ticket ingestion depending on collection name."""
    col_name = store.collection_name.lower()
    if "table_aware" in col_name:
        return ingest_articles(store, strategy="table_aware", reset=reset)
    elif "paragraph" in col_name or "article" in col_name:
        return ingest_articles(store, strategy="paragraph", reset=reset)
    else:
        return run_ingest(store, reset=reset)


@router.post("/ingest", response_model=IngestStats)
def ingest_documents(
    request: IngestRequest,
    store: VectorStore = Depends(get_vector_store),
    retriever: Retriever = Depends(get_retriever),
) -> IngestStats:
    """(Re)index everything in `documents/`.

    Idempotent — chunk ids are derived from ticket/article ids.
    """
    try:
        stats = _run_appropriate_ingest(store, reset=request.reset)
        retriever.reset_bm25_index()
        return stats
    except FileNotFoundError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    except Exception as exc:
        logger.exception("ingest failed")
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"Ingest failed: {exc}") from exc


@router.post("/ingest/upload", response_model=IngestStats)
async def upload_documents(
    files: list[UploadFile] = File(...),
    store: VectorStore = Depends(get_vector_store),
    retriever: Retriever = Depends(get_retriever),
) -> IngestStats:
    """Save uploaded ticket/article files into `documents/` and reindex."""
    settings = get_settings()
    documents_dir = settings.documents_dir
    documents_dir.mkdir(parents=True, exist_ok=True)
    help_articles_dir = documents_dir / "help_articles"
    help_articles_dir.mkdir(parents=True, exist_ok=True)

    saved: list[Path] = []
    has_md = False
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

        if suffix == ".md":
            destination = help_articles_dir / name
            has_md = True
        else:
            destination = documents_dir / name

        destination.write_bytes(content)
        saved.append(destination)
        logger.info("saved upload %s (%d bytes)", destination.name, len(content))

    if not saved:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "No files were uploaded.")

    stats = _run_appropriate_ingest(store, reset=False)
    retriever.reset_bm25_index()
    return stats


@router.delete("/tickets/{ticket_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_ticket(
    ticket_id: str,
    store: VectorStore = Depends(get_vector_store),
    retriever: Retriever = Depends(get_retriever),
) -> None:
    """Remove a ticket/article's chunks from the index."""
    store.delete_ticket(ticket_id)
    retriever.reset_bm25_index()
    logger.info("deleted chunks for ticket %s", ticket_id)


@router.delete("/documents/{doc_id}", status_code=status.HTTP_200_OK)
@router.delete("/ingest/documents/{doc_id}", status_code=status.HTTP_200_OK)
def delete_document(
    doc_id: str,
    store: VectorStore = Depends(get_vector_store),
    retriever: Retriever = Depends(get_retriever),
) -> dict[str, Any]:
    """Delete a document file from disk, wipe its chunks from Chroma DB, and reset BM25."""
    settings = get_settings()
    deleted_files = []
    
    # 1. Search and delete physical file on disk
    search_dirs = [settings.documents_dir / "help_articles", settings.documents_dir]
    for d in search_dirs:
        if d.exists():
            for p in d.glob(f"*{doc_id}*"):
                if p.is_file():
                    try:
                        p.unlink()
                        deleted_files.append(p.name)
                        logger.info("deleted physical file %s", p)
                    except Exception as exc:
                        logger.warning("failed to delete file %s: %s", p, exc)

    # 2. Delete chunks from Chroma DB
    store.delete_ticket(doc_id)

    # 3. Reset BM25 index
    retriever.reset_bm25_index()

    return {
        "status": "success",
        "doc_id": doc_id,
        "deleted_files": deleted_files,
        "message": f"Successfully removed {doc_id} from database and disk.",
    }


@router.delete("/images/{filename}", status_code=status.HTTP_200_OK)
@router.delete("/ingest/images/{filename}", status_code=status.HTTP_200_OK)
def delete_image(filename: str) -> dict[str, Any]:
    """Delete an uploaded screenshot image from documents/uploaded_images/."""
    settings = get_settings()
    safe_name = SAFE_NAME.sub("_", Path(filename).name)
    img_path = settings.documents_dir / "uploaded_images" / safe_name

    if img_path.exists() and img_path.is_file():
        img_path.unlink()
        logger.info("deleted uploaded image %s", img_path)
        return {"status": "success", "filename": safe_name, "message": f"Deleted {safe_name}"}

    raise HTTPException(status.HTTP_404_NOT_FOUND, f"Image {safe_name} not found.")
