"""
Pydantic Schemas for KnowledgeForge AI.
Defines strongly-typed request and response payloads for the FastAPI backend.
"""
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """System health check model."""
    status: str = "ok"
    version: str = "1.0.0"
    provider: str
    environment: str
    total_documents: int = 0
    total_indexed_chunks: int = 0


class DocumentUploadResponse(BaseModel):
    """Response returned upon successful file ingestion."""
    document_id: str
    filename: str
    file_type: str
    pages_count: int
    chunks_count: int
    status: str = "indexed"


class DocumentMetadata(BaseModel):
    """Metadata representing an indexed document."""
    document_id: str
    filename: str
    file_type: str
    size_bytes: int
    chunks_count: int
    uploaded_at: str


class DocumentListResponse(BaseModel):
    """Summary of active knowledge base documents."""
    documents: List[DocumentMetadata]
    total_documents: int
    total_chunks: int


class DeleteDocumentResponse(BaseModel):
    """Confirmation of document removal."""
    document_id: str
    filename: str
    deleted_chunks: int
    status: str = "deleted"


class QueryRequest(BaseModel):
    """Input payload for grounded RAG query."""
    query: str = Field(..., min_length=2, max_length=1500, description="User question or research query")
    document_ids: Optional[List[str]] = Field(default=None, description="Optional document IDs for scoped retrieval")
    top_k: int = Field(default=4, ge=1, le=10, description="Maximum chunks to retrieve")
    similarity_threshold: float = Field(default=0.05, ge=0.0, le=1.0, description="Minimum similarity cut-off")
    enable_rerank: bool = Field(default=True, description="Enable BM25-density hybrid reranking")
    session_id: Optional[str] = Field(default="default_session", description="Conversation session identifier")


class SourceCitation(BaseModel):
    """Structured citation item."""
    document_id: str
    document_name: str
    page_number: int
    chunk_id: str
    score: float
    excerpt: str


class QueryResponse(BaseModel):
    """Comprehensive grounded RAG response."""
    answer: str
    grounded: bool
    sources: List[SourceCitation]
    rewritten_query: Optional[str] = None
    diagnostics: Dict[str, Any] = Field(default_factory=dict)
    session_id: str
