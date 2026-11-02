"""
Automated Unit and Integration Tests for Intermediate Level: DocuMind AI.
Tests Document Ingestion, Recursive Chunking, FAISS Vector Indexing,
Similarity Search, Grounded QA Generation, and Insufficient Evidence Handling.
"""
import pytest
import os
import sys

# Ensure root directory is on sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from intermediate.ingestion.loader import DocumentLoader, DocumentLoadingError
from intermediate.ingestion.chunker import TextChunker
from intermediate.retrieval.embeddings import EmbeddingService, EmbeddingError
from intermediate.retrieval.vector_store import FAISSVectorStore
from intermediate.generation.qa_chain import GroundedQAChain, RAGGenerationError
from beginner.services.llm_service import LLMService


# ==============================================================================
# 1. INGESTION & DOCUMENT LOADER TESTS
# ==============================================================================
def test_load_valid_txt():
    sample_text = b"This is a test document explaining the fundamental properties of neural networks."
    pages = DocumentLoader.load_file(sample_text, "neural_nets.txt")
    assert len(pages) == 1
    assert pages[0]["page_number"] == 1
    assert pages[0]["source"] == "neural_nets.txt"
    assert "neural networks" in pages[0]["text"]


def test_load_empty_file_fails():
    with pytest.raises(DocumentLoadingError, match="completely empty"):
        DocumentLoader.load_file(b"", "empty.txt")


def test_load_unsupported_format_fails():
    with pytest.raises(DocumentLoadingError, match="Unsupported file format"):
        DocumentLoader.load_file(b"some content", "data.docx")


def test_load_whitespace_txt_fails():
    with pytest.raises(DocumentLoadingError, match="contains no readable text"):
        DocumentLoader.load_file(b"   \n\t   ", "blank.txt")


# ==============================================================================
# 2. CHUNKING & METADATA PRESERVATION TESTS
# ==============================================================================
def test_chunker_sliding_window_metadata():
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    pages = [
        {"page_number": 1, "source": "guide.pdf", "text": "A" * 150},
        {"page_number": 2, "source": "guide.pdf", "text": "B" * 120}
    ]
    chunks = chunker.split_pages(pages)

    assert len(chunks) > 2
    # Verify metadata fields
    for c in chunks:
        assert "chunk_id" in c
        assert "source" in c
        assert "page_number" in c
        assert "char_count" in c
        assert c["source"] == "guide.pdf"

    # Check that page 1 and page 2 are represented
    pages_found = {c["page_number"] for c in chunks}
    assert 1 in pages_found
    assert 2 in pages_found


def test_chunker_invalid_hyperparameters():
    with pytest.raises(ValueError, match="strictly less than chunk_size"):
        TextChunker(chunk_size=100, chunk_overlap=100)

    with pytest.raises(ValueError, match="cannot be negative"):
        TextChunker(chunk_size=100, chunk_overlap=-5)


# ==============================================================================
# 3. EMBEDDING & FAISS VECTOR STORE TESTS
# ==============================================================================
def test_embedding_service_mock_mode():
    embedder = EmbeddingService(use_mock=True)
    vec = embedder.embed_text("Artificial Intelligence and Machine Learning")
    assert isinstance(vec, list)
    assert len(vec) == 384
    assert all(isinstance(x, float) for x in vec)


def test_embedding_empty_text_fails():
    embedder = EmbeddingService(use_mock=True)
    with pytest.raises(ValueError, match="Cannot embed empty text"):
        embedder.embed_text("")


def test_faiss_vector_store_operations():
    embedder = EmbeddingService(use_mock=True)
    store = FAISSVectorStore(dimension=384)

    chunks = [
        {"chunk_id": "c1", "source": "docA.txt", "page_number": 1, "text": "PostgreSQL is a relational database management system."},
        {"chunk_id": "c2", "source": "docA.txt", "page_number": 2, "text": "Redis is an in-memory key-value data structure store."},
        {"chunk_id": "c3", "source": "docB.txt", "page_number": 1, "text": "Kubernetes automates software deployment and scaling."}
    ]
    texts = [c["text"] for c in chunks]
    embeddings = embedder.embed_documents(texts)

    # Ingest
    total = store.add_documents(chunks, embeddings)
    assert total == 3
    assert store.total_chunks == 3

    # Query search
    q_vec = embedder.embed_text("relational database SQL queries")
    results = store.similarity_search(query_vector=q_vec, top_k=2)

    assert len(results) <= 2
    assert "similarity_score" in results[0]
    assert results[0]["similarity_score"] >= -1.0

    # Inspection
    docs = store.get_indexed_documents()
    assert "docA.txt" in docs
    assert "docB.txt" in docs

    # Deletion
    removed = store.remove_document("docB.txt")
    assert removed == 1
    assert "docB.txt" not in store.get_indexed_documents()

    # Clear
    store.clear()
    assert store.total_chunks == 0


def test_faiss_empty_search():
    store = FAISSVectorStore(dimension=384)
    results = store.similarity_search(query_vector=[0.1] * 384, top_k=3)
    assert results == []


# ==============================================================================
# 4. GROUNDED QA GENERATION & INSUFFICIENT EVIDENCE TESTS
# ==============================================================================
def test_grounded_qa_with_evidence():
    llm = LLMService(use_mock=True)
    qa = GroundedQAChain(llm_service=llm)

    chunks = [
        {
            "chunk_id": "c1",
            "source": "sys_specs.pdf",
            "page_number": 4,
            "similarity_score": 0.88,
            "text": "The distributed ledger uses Byzantine Fault Tolerance with 3f+1 nodes."
        }
    ]

    res = qa.answer_question("How many nodes are needed for consensus?", retrieved_chunks=chunks)
    assert res["status"] == "success"
    assert res["grounded"] is True
    assert len(res["sources"]) == 1
    assert res["sources"][0]["source"] == "sys_specs.pdf"
    assert res["sources"][0]["page_number"] == 4


def test_grounded_qa_insufficient_evidence_when_empty():
    llm = LLMService(use_mock=True)
    qa = GroundedQAChain(llm_service=llm)

    # Empty chunks passed
    res = qa.answer_question("What is the speed of light in vacuum?", retrieved_chunks=[])
    assert res["status"] == "insufficient_evidence"
    assert res["grounded"] is False
    assert "insufficient evidence" in res["answer"].lower()
    assert res["sources"] == []


def test_grounded_qa_empty_query_fails():
    llm = LLMService(use_mock=True)
    qa = GroundedQAChain(llm_service=llm)

    with pytest.raises(ValueError, match="Question cannot be empty"):
        qa.answer_question("", retrieved_chunks=[])
