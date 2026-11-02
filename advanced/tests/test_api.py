"""
Automated Tests for FastAPI REST Endpoints.
"""
import pytest
import os
import sys
from fastapi.testclient import TestClient

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from advanced.backend.main import app
from advanced.backend.services.rag_service import RAGService
from advanced.backend.api.routes import set_rag_service


@pytest.fixture(autouse=True)
def setup_mock_service():
    """Injects a clean RAGService in deterministic test mode before each test."""
    mock_service = RAGService(use_mock=True)
    set_rag_service(mock_service)
    yield mock_service


@pytest.fixture
def client():
    return TestClient(app)


def test_health_endpoint(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ok"
    assert "version" in data
    assert "total_documents" in data


def test_document_lifecycle_endpoints(client):
    # 1. Ingest document via upload
    file_content = b"# Distributed Consensus\nRaft achieves consensus via leader election and log replication."
    files = {"file": ("raft.md", file_content, "text/markdown")}
    data = {"chunk_size": "200", "chunk_overlap": "20"}

    up_res = client.post("/documents/upload", files=files, data=data)
    assert up_res.status_code == 201
    up_data = up_res.json()
    doc_id = up_data["document_id"]
    assert doc_id is not None
    assert up_data["filename"] == "raft.md"

    # 2. List documents
    list_res = client.get("/documents")
    assert list_res.status_code == 200
    list_data = list_res.json()
    assert list_data["total_documents"] >= 1
    assert any(d["document_id"] == doc_id for d in list_data["documents"])

    # 3. Query document
    q_payload = {
        "query": "How does Raft achieve consensus?",
        "document_ids": [doc_id],
        "top_k": 3,
        "similarity_threshold": 0.0,
        "enable_rerank": True
    }
    q_res = client.post("/query", json=q_payload)
    assert q_res.status_code == 200
    q_data = q_res.json()
    assert "answer" in q_data
    assert "diagnostics" in q_data
    assert q_data["grounded"] is True

    # 4. Delete document
    del_res = client.delete(f"/documents/{doc_id}")
    assert del_res.status_code == 200
    del_data = del_res.json()
    assert del_data["status"] == "deleted"

    # 5. Delete non-existent returns 404
    del_404 = client.delete("/documents/non_existent_id_999")
    assert del_404.status_code == 404


def test_query_validation_error(client):
    # Query too short (<2 chars)
    invalid_q = {"query": "a"}
    res = client.post("/query", json=invalid_q)
    assert res.status_code == 422


def test_session_endpoint(client):
    res = client.get("/sessions/test_session_123")
    assert res.status_code == 200
    assert res.json()["session_id"] == "test_session_123"
