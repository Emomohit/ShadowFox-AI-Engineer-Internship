"""
Production FAISS Vector Index with Document-Scoped Filtering.
"""
from typing import List, Dict, Any, Optional
import os
import pickle
import numpy as np
import faiss
from advanced.backend.core.exceptions import VectorIndexError


class DocumentScopedFAISSIndex:
    """FAISS vector index with document ID partitioning and scope filtering."""

    def __init__(self, dimension: int = 384, storage_path: Optional[str] = None):
        self.dimension = dimension
        self.storage_path = storage_path
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks: List[Dict[str, Any]] = []
        self.documents_registry: Dict[str, Dict[str, Any]] = {}

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (arr / norms).astype(np.float32)

    def add_document_chunks(
        self,
        document_id: str,
        document_metadata: Dict[str, Any],
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]]
    ) -> int:
        """Adds chunks from a specific document and indexes their vector embeddings."""
        if not chunks or not embeddings:
            return 0

        if len(chunks) != len(embeddings):
            raise VectorIndexError(f"Chunks count ({len(chunks)}) != embeddings count ({len(embeddings)})")

        raw_mat = np.array(embeddings, dtype=np.float32)
        if raw_mat.shape[1] != self.dimension:
            if self.index.ntotal == 0:
                self.dimension = raw_mat.shape[1]
                self.index = faiss.IndexFlatIP(self.dimension)
            else:
                raise VectorIndexError(
                    f"Dimension mismatch: expected {self.dimension}, got {raw_mat.shape[1]}"
                )

        norm_mat = self._normalize(raw_mat)
        self.index.add(norm_mat)
        self.chunks.extend(chunks)
        self.documents_registry[document_id] = document_metadata
        return len(chunks)

    def search(
        self,
        query_vector: List[float],
        top_k: int = 4,
        score_threshold: float = 0.0,
        allowed_document_ids: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes similarity search with optional document-scoping and score thresholds.
        """
        if self.index.ntotal == 0 or not self.chunks:
            return []

        q_mat = np.array([query_vector], dtype=np.float32)
        if q_mat.shape[1] != self.dimension:
            raise VectorIndexError(f"Query dimension {q_mat.shape[1]} != index {self.dimension}")

        norm_q = self._normalize(q_mat)

        # Retrieve a slightly wider pool if document scoping is applied
        fetch_k = min(self.index.ntotal, top_k * 3 if allowed_document_ids else top_k)
        scores, indices = self.index.search(norm_q, fetch_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx]

            # Apply document scoping filter
            if allowed_document_ids and chunk.get("document_id") not in allowed_document_ids:
                continue

            if score >= score_threshold:
                c = dict(chunk)
                c["similarity_score"] = float(round(score, 4))
                results.append(c)

            if len(results) >= top_k:
                break

        return results

    def delete_document(self, document_id: str) -> int:
        """Deletes all chunks belonging to document_id and rebuilds the FAISS index."""
        if document_id not in self.documents_registry and not any(c.get("document_id") == document_id for c in self.chunks):
            return 0

        remaining = [c for c in self.chunks if c.get("document_id") != document_id]
        deleted_count = len(self.chunks) - len(remaining)
        self.chunks = remaining
        self.documents_registry.pop(document_id, None)

        # Re-initialize index structure
        self.index = faiss.IndexFlatIP(self.dimension)
        return deleted_count

    def clear(self) -> None:
        """Resets the vector index and registry."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks.clear()
        self.documents_registry.clear()

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)

    @property
    def total_documents(self) -> int:
        return len(self.documents_registry)
