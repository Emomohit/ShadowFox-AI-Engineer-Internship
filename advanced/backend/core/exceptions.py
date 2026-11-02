"""
Custom Exceptions for KnowledgeForge AI.
"""


class KnowledgeForgeException(Exception):
    """Base exception for all system errors."""
    pass


class IngestionError(KnowledgeForgeException):
    """Raised when file parsing, validation, or chunking fails."""
    pass


class EmbeddingServiceError(KnowledgeForgeException):
    """Raised when vector embedding generation fails."""
    pass


class VectorIndexError(KnowledgeForgeException):
    """Raised when FAISS operations fail."""
    pass


class WorkflowExecutionError(KnowledgeForgeException):
    """Raised when LangGraph workflow execution fails."""
    pass
