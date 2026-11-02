"""
Workflow State Definitions for KnowledgeForge AI.
"""
from typing import TypedDict, List, Dict, Any, Optional


class RAGState(TypedDict):
    """Encapsulates the explicit state traversed across LangGraph nodes."""
    raw_query: str
    rewritten_query: str
    document_ids: Optional[List[str]]
    top_k: int
    similarity_threshold: float
    enable_rerank: bool
    retrieved_chunks: List[Dict[str, Any]]
    reranked_chunks: List[Dict[str, Any]]
    answer: str
    grounded: bool
    grounding_score: float
    grounding_reason: str
    sources: List[Dict[str, Any]]
    diagnostics: Dict[str, Any]
    retry_count: int
    error: Optional[str]
