"""
Automated Tests for Advanced Ingestion: MultiFormatIngestor.
"""
import pytest
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from advanced.backend.rag.ingestion import MultiFormatIngestor
from advanced.backend.core.exceptions import IngestionError


def test_ingest_markdown_file():
    md_content = b"# Architecture\n\nThis is a markdown document detailing distributed caching systems."
    doc_id, chunks = MultiFormatIngestor.process_file(
        file_bytes=md_content,
        filename="system_design.md",
        chunk_size=100,
        chunk_overlap=20
    )
    assert len(doc_id) == 16
    assert len(chunks) >= 1
    assert chunks[0]["document_name"] == "system_design.md"
    assert "distributed caching" in chunks[0]["text"]


def test_ingest_txt_file():
    txt_content = b"Microservices communicate via gRPC and message brokers."
    doc_id, chunks = MultiFormatIngestor.process_file(
        file_bytes=txt_content,
        filename="microservices.txt"
    )
    assert len(doc_id) == 16
    assert chunks[0]["document_id"] == doc_id


def test_ingest_empty_file_rejected():
    with pytest.raises(IngestionError, match="empty"):
        MultiFormatIngestor.process_file(b"", "empty.md")


def test_ingest_unsupported_extension_rejected():
    with pytest.raises(IngestionError, match="Unsupported format"):
        MultiFormatIngestor.process_file(b"some content", "data.json")


def test_stable_hashing_consistency():
    content = b"Deterministic payload for cryptographic SHA256 integrity verification."
    id1, _ = MultiFormatIngestor.process_file(content, "fileA.txt")
    id2, _ = MultiFormatIngestor.process_file(content, "fileB.txt")
    # Same content must produce identical document IDs
    assert id1 == id2
