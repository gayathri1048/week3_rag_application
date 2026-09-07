"""Route modules, aggregated into one router mounted at /api."""

from fastapi import APIRouter

from . import chat, health, ingest, inspection, tickets

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(chat.router, tags=["chat"])
api_router.include_router(tickets.router, tags=["tickets"])
api_router.include_router(ingest.router, tags=["ingest"])
api_router.include_router(inspection.router, tags=["inspection"])

__all__ = ["api_router"]
