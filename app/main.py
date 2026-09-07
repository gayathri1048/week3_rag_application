"""FastAPI application entry point.

    uvicorn app.main:app --reload --port 8000
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .api.routes import api_router
from .config import get_settings
from .dependencies import get_vector_store

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Warm the vector store on boot so the first request isn't slow.

    Failures here are logged, not fatal — the app should still start so /health can
    report what's wrong and the frontend can show a setup banner.
    """
    settings = get_settings()
    logging.basicConfig(
        level=settings.log_level.upper(),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )

    try:
        store = get_vector_store()
        count = store.count()
        logger.info("vector store ready: '%s' holds %d chunks", store.collection_name, count)
        if count == 0:
            logger.warning(
                "index is empty — POST /api/ingest or run `python -m app.ingestion.pipeline`"
            )
    except Exception:
        logger.exception("vector store failed to initialise")

    yield
    logger.info("shutting down")


app = FastAPI(
    title="Customer Support Tickets — RAG API",
    description=(
        "Retrieval-augmented Q&A, triage, and reply drafting over a support ticket "
        "archive. Retrieval runs locally via Chroma; generation runs on Claude."
    ),
    version=__version__,
    lifespan=lifespan,
)

settings = get_settings()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api")

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return JSON (not an HTML traceback page) so the frontend can always parse errors."""
    logger.exception("unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error", "path": request.url.path},
    )


@app.get("/", include_in_schema=False)
@app.get("/ui", include_in_schema=False)
def root_ui():
    index_file = STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {
        "name": "Customer Support Tickets RAG API",
        "version": __version__,
        "docs": "/docs",
        "health": "/api/health",
    }
