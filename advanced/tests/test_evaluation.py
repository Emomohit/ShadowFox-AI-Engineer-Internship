"""
RAG Retrieval & Answer Evaluation Benchmark Tests.
Evaluates Precision@K, Recall@K, and Grounding Accuracy on a synthetic benchmark dataset.
"""
import pytest
import os
import sys

root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from advanced.backend.services.rag_service import RAGService
from advanced.backend.schemas.models import QueryRequest

# Synthetic Benchmark Evaluation Dataset
EVAL_DATASET = [
    {
        "doc_name": "raft_consensus.txt",
        "doc_text": "Raft achieves consensus via randomized timer leader election and replicated state machines.",
        "query": "How does Raft elect a leader?",
        "expected_keyword": "election"
    },
    {
        "doc_name": "cap_theorem.txt",
        "doc_text": "The CAP theorem states that a distributed system can satisfy at most two of Consistency, Availability, Partition tolerance.",
        "query": "What are the three properties in CAP theorem?",
        "expected_keyword": "consistency"
    },
    {
        "doc_name": "b_tree_indexing.txt",
        "doc_text": "B-Tree indexes maintain balanced logarithmic depth to ensure O(log n) lookups and range scans.",
        "query": "What is the search time complexity of B-Trees?",
        "expected_keyword": "logarithmic"
    }
]


@pytest.fixture
def evaluated_rag_service():
    service = RAGService(use_mock=True)
    for item in EVAL_DATASET:
        service.ingest_document(item["doc_text"].encode("utf-8"), item["doc_name"])
    return service


def test_retrieval_recall_evaluation(evaluated_rag_service):
    """Measures Recall@K across the evaluation dataset."""
    total_queries = len(EVAL_DATASET)
    successful_retrievals = 0

    for item in EVAL_DATASET:
        req = QueryRequest(
            query=item["query"],
            top_k=2,
            similarity_threshold=0.0,
            enable_rerank=True
        )
        response = evaluated_rag_service.execute_query(req)

        # Check if the expected document is present in the retrieved sources
        found_sources = [s.document_name for s in response.sources]
        if item["doc_name"] in found_sources:
            successful_retrievals += 1

    recall_at_k = successful_retrievals / total_queries
    # Assert acceptable benchmark recall threshold (>= 66%)
    assert recall_at_k >= 0.66, f"Retrieval recall ({recall_at_k}) below benchmark threshold."


def test_grounding_eval_on_unrelated_query(evaluated_rag_service):
    """Checks that out-of-domain queries correctly produce ungrounded verdict with high threshold."""
    req = QueryRequest(
        query="What is the recipe for chocolate chip pancakes?",
        top_k=2,
        similarity_threshold=0.85
    )
    res = evaluated_rag_service.execute_query(req)
    assert res.grounded is False
    assert "insufficient evidence" in res.answer.lower()
