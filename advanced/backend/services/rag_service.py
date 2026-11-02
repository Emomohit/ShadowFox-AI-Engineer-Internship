"""
High-Level Production RAG Orchestrator Service for KnowledgeForge AI.
"""
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone
from advanced.backend.core.config import settings
from advanced.backend.rag.ingestion import MultiFormatIngestor
from advanced.backend.rag.embeddings import ProductionEmbeddingService
from advanced.backend.rag.vector_index import DocumentScopedFAISSIndex
from advanced.backend.rag.reranker import HybridReranker
from advanced.backend.workflows.graph import RAGWorkflowBuilder
from beginner.services.llm_service import LLMService
from advanced.backend.schemas.models import (
    DocumentMetadata,
    QueryRequest,
    QueryResponse,
    SourceCitation
)


class RAGService:
    """Singleton-style or injected service managing the end-to-end RAG lifecycle."""

    def __init__(self, use_mock: bool = False):
        self.embedding_service = ProductionEmbeddingService(use_mock=use_mock)
        self.vector_index = DocumentScopedFAISSIndex(dimension=384)
        self.reranker = HybridReranker()
        self.llm_service = LLMService(use_mock=use_mock)

        # Build and compile LangGraph
        self.workflow_builder = RAGWorkflowBuilder(
            vector_index=self.vector_index,
            embedding_service=self.embedding_service,
            llm_service=self.llm_service,
            reranker=self.reranker
        )
        self.app = self.workflow_builder.build()

    def ingest_document(
        self,
        file_bytes: bytes,
        filename: str,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ) -> Dict[str, Any]:
        """Processes, embeds, and indexes a document."""
        # 1. Ingest & Chunk
        doc_id, chunks = MultiFormatIngestor.process_file(
            file_bytes=file_bytes,
            filename=filename,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            max_size_bytes=settings.MAX_FILE_SIZE_BYTES
        )

        # 2. Embed
        texts = [c["text"] for c in chunks]
        embeddings = self.embedding_service.embed_documents(texts)

        # 3. Index into FAISS
        meta = {
            "document_id": doc_id,
            "filename": filename,
            "file_type": filename.split(".")[-1].lower(),
            "size_bytes": len(file_bytes),
            "chunks_count": len(chunks),
            "uploaded_at": datetime.now(timezone.utc).isoformat()
        }
        self.vector_index.add_document_chunks(
            document_id=doc_id,
            document_metadata=meta,
            chunks=chunks,
            embeddings=embeddings
        )

        return {
            "document_id": doc_id,
            "filename": filename,
            "file_type": meta["file_type"],
            "pages_count": len(set(c["page_number"] for c in chunks)),
            "chunks_count": len(chunks),
            "status": "indexed"
        }

    def list_documents(self) -> List[DocumentMetadata]:
        """Lists metadata of all indexed documents."""
        docs = []
        for doc_id, meta in self.vector_index.documents_registry.items():
            docs.append(DocumentMetadata(**meta))
        return docs

    def delete_document(self, document_id: str) -> Dict[str, Any]:
        """Deletes a document from the FAISS vector index."""
        filename = self.vector_index.documents_registry.get(document_id, {}).get("filename", "Unknown")
        deleted_count = self.vector_index.delete_document(document_id)
        return {
            "document_id": document_id,
            "filename": filename,
            "deleted_chunks": deleted_count,
            "status": "deleted" if deleted_count > 0 else "not_found"
        }

    def execute_query(self, req: QueryRequest) -> QueryResponse:
        """Executes a research query through the LangGraph state machine."""
        initial_state = {
            "raw_query": req.query,
            "rewritten_query": req.query,
            "document_ids": req.document_ids,
            "top_k": req.top_k,
            "similarity_threshold": req.similarity_threshold,
            "enable_rerank": req.enable_rerank,
            "retrieved_chunks": [],
            "reranked_chunks": [],
            "answer": "",
            "grounded": False,
            "grounding_score": 0.0,
            "grounding_reason": "",
            "sources": [],
            "diagnostics": {},
            "retry_count": 0,
            "error": None
        }

        # Run through LangGraph
        final_state = self.app.invoke(initial_state)

        # Format source citations
        citations = []
        for s in final_state.get("sources", []):
            citations.append(SourceCitation(**s))

        diagnostics = {
            "retrieved_count": len(final_state.get("retrieved_chunks", [])),
            "reranked_count": len(final_state.get("reranked_chunks", [])),
            "grounding_score": final_state.get("grounding_score", 0.0),
            "grounding_reason": final_state.get("grounding_reason", ""),
            "rewritten_query": final_state.get("rewritten_query")
        }

        return QueryResponse(
            answer=final_state.get("answer", "No response generated."),
            grounded=final_state.get("grounded", False),
            sources=citations,
            rewritten_query=final_state.get("rewritten_query"),
            diagnostics=diagnostics,
            session_id=req.session_id or "default_session"
        )
