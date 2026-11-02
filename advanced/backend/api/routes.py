"""
FastAPI REST API Routes for KnowledgeForge AI.
"""
from typing import Optional, List
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status, Depends
from advanced.backend.schemas.models import (
    HealthResponse,
    DocumentUploadResponse,
    DocumentListResponse,
    DeleteDocumentResponse,
    QueryRequest,
    QueryResponse
)
from advanced.backend.services.rag_service import RAGService
from advanced.backend.core.config import settings
from advanced.backend.core.exceptions import IngestionError, KnowledgeForgeException

router = APIRouter()

# Global RAG service dependency (allows test injection or mock override)
_rag_service_instance: Optional[RAGService] = None


def get_rag_service() -> RAGService:
    global _rag_service_instance
    if _rag_service_instance is None:
        _rag_service_instance = RAGService(use_mock=(settings.LLM_PROVIDER == "mock" or not settings.GEMINI_API_KEY))
    return _rag_service_instance


def set_rag_service(service: RAGService) -> None:
    """Helper to inject custom or mocked service in tests."""
    global _rag_service_instance
    _rag_service_instance = service


@router.get("/health", response_model=HealthResponse, summary="System Health Status")
async def health_check(service: RAGService = Depends(get_rag_service)):
    """Returns application health, version, provider, and current index statistics."""
    return HealthResponse(
        status="ok",
        version=settings.APP_VERSION,
        provider=settings.LLM_PROVIDER,
        environment=settings.APP_ENV,
        total_documents=service.vector_index.total_documents,
        total_indexed_chunks=service.vector_index.total_chunks
    )


@router.post("/documents/upload", response_model=DocumentUploadResponse, status_code=status.HTTP_201_CREATED, summary="Upload & Ingest Document")
async def upload_document(
    file: UploadFile = File(...),
    chunk_size: int = Form(default=500),
    chunk_overlap: int = Form(default=50),
    service: RAGService = Depends(get_rag_service)
):
    """
    Ingests PDF, TXT, or Markdown documents, extracts text, generates chunk embeddings,
    and stores them in the FAISS vector index.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename cannot be empty.")

    try:
        content = await file.read()
        res = service.ingest_document(
            file_bytes=content,
            filename=file.filename,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return DocumentUploadResponse(**res)
    except IngestionError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@router.get("/documents", response_model=DocumentListResponse, summary="List Active Documents")
async def list_documents(service: RAGService = Depends(get_rag_service)):
    """Retrieves all indexed documents in the active knowledge store."""
    docs = service.list_documents()
    return DocumentListResponse(
        documents=docs,
        total_documents=len(docs),
        total_chunks=service.vector_index.total_chunks
    )


@router.delete("/documents/{document_id}", response_model=DeleteDocumentResponse, summary="Delete Document")
async def delete_document(document_id: str, service: RAGService = Depends(get_rag_service)):
    """Removes a document and its chunks from the FAISS vector index."""
    res = service.delete_document(document_id)
    if res["status"] == "not_found":
        raise HTTPException(status_code=404, detail=f"Document ID '{document_id}' not found.")
    return DeleteDocumentResponse(**res)


@router.post("/query", response_model=QueryResponse, summary="Execute Grounded RAG Query")
async def query_knowledge_base(
    request: QueryRequest,
    service: RAGService = Depends(get_rag_service)
):
    """
    Orchestrates an agentic multi-step RAG workflow via LangGraph:
    Validate -> Rewrite -> FAISS Retrieve -> Hybrid Rerank -> Generate -> Grounding Check.
    """
    try:
        response = service.execute_query(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query workflow failed: {str(e)}")


@router.get("/sessions/{session_id}", summary="Get Session Info")
async def get_session(session_id: str):
    """Returns conversation session status."""
    return {
        "session_id": session_id,
        "status": "active",
        "description": "Stateless LangGraph execution context"
    }
