"""
KnowledgeForge AI Application Entrypoint.
Exposes the FastAPI application instance for Uvicorn and ASGI servers.
"""
from advanced.backend.main import app

__all__ = ["app"]
