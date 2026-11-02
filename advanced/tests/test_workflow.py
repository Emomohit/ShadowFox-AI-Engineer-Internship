"""
Automated Tests for LangGraph Multi-Step Workflow.
"""
import pytest
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from advanced.backend.services.rag_service import RAGService
from advanced.backend.schemas.models import QueryRequest
from advanced.backend.rag.reranker import HybridReranker


@pytest.fixture
def rag_service():
    service = RAGService(use_mock=True)
    # Ingest two distinct test documents
    doc1 = b"PostgreSQL provides advanced relational storage, ACID transactions, and MVCC concurrency."
    doc2 = b"Kafka is a distributed event streaming platform capable of handling trillions of events a day."
    service.ingest_document(doc1, "postgres.txt")
    service.ingest_document(doc2, "kafka.txt")
    return service


def test_document_scoped_query(rag_service):
    # Find postgres document_id
    docs = rag_service.list_documents()
    postgres_id = next(d.document_id for d in docs if d.filename == "postgres.txt")
    kafka_id = next(d.document_id for d in docs if d.filename == "kafka.txt")

    # Query scoped strictly to postgres_id
    req = QueryRequest(
        query="Explain event streaming and event logs",
        document_ids=[postgres_id],
        top_k=3
    )
    resp = rag_service.execute_query(req)

    # All returned sources MUST belong to postgres_id, never kafka_id
    for src in resp.sources:
        assert src.document_id == postgres_id
        assert src.document_id != kafka_id


def test_workflow_insufficient_evidence_branch(rag_service):
    # Reset vector store to force empty retrieval
    rag_service.vector_index.clear()

    req = QueryRequest(
        query="What is the quantum spin of a Higgs boson?",
        top_k=3,
        similarity_threshold=0.5
    )
    resp = rag_service.execute_query(req)

    assert resp.grounded is False
    assert "insufficient evidence" in resp.answer.lower()
    assert len(resp.sources) == 0


def test_hybrid_reranker_logic():
    reranker = HybridReranker()
    candidates = [
        {"chunk_id": "c1", "similarity_score": 0.80, "text": "General architecture notes with overview."},
        {"chunk_id": "c2", "similarity_score": 0.75, "text": "PostgreSQL ACID transactions and relational locks."}
    ]
    query = "PostgreSQL ACID"
    reranked = reranker.rerank(query=query, candidates=candidates, top_k=2)

    # c2 has lower initial similarity but high lexical overlap with query -> boosted to top
    assert reranked[0]["chunk_id"] == "c2"
    assert "rerank_score" in reranked[0]
