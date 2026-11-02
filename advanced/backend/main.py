"""
KnowledgeForge AI - FastAPI Production Application Entrypoint.
"""
import sys
import os

# Ensure root directory is accessible on sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from advanced.backend.core.config import settings
from advanced.backend.api.routes import router
from advanced.backend.core.exceptions import KnowledgeForgeException

app = FastAPI(
    title="KnowledgeForge AI",
    description=(
        "Production-style Retrieval-Augmented Generation (RAG) API featuring "
        "LangGraph multi-step orchestration, FAISS vector indexing, hybrid reranking, "
        "and strict evidence-grounded synthesis."
    ),
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for Streamlit frontend or web dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Centralized Exception Handler
@app.exception_handler(KnowledgeForgeException)
async def custom_exception_handler(request: Request, exc: KnowledgeForgeException):
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content={"error_type": exc.__class__.__name__, "message": str(exc)}
    )

# Include API Routes
app.include_router(router, prefix="/api/v1")
app.include_router(router)  # also expose top-level for convenience (/health, /query, etc.)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("advanced.backend.main:app", host="0.0.0.0", port=8000, reload=True)
