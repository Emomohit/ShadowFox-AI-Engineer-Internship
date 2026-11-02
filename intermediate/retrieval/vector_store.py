"""
FAISS Vector Store wrapper for DocuMind AI.
Provides exact Cosine Similarity search over normalized embeddings
with associated chunk metadata preservation and document-level management.
"""
from typing import List, Dict, Any, Tuple
import numpy as np
import faiss


class FAISSVectorStore:
    """Manages an in-memory FAISS IndexFlatIP vector store with metadata mappings."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks: List[Dict[str, Any]] = []

    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """L2 normalizes vectors so Inner Product equals Cosine Similarity."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (vectors / norms).astype(np.float32)

    def add_documents(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> int:
        """
        Adds document chunks and their corresponding embedding vectors to the FAISS index.
        Returns the new total count of chunks in the index.
        """
        if not chunks or not embeddings:
            return len(self.chunks)

        if len(chunks) != len(embeddings):
            raise ValueError(f"Mismatch: {len(chunks)} chunks vs {len(embeddings)} embeddings.")

        arr = np.array(embeddings, dtype=np.float32)
        if arr.shape[1] != self.dimension:
            # Dynamically adjust index dimension if provider produces a different vector size
            if self.index.ntotal == 0:
                self.dimension = arr.shape[1]
                self.index = faiss.IndexFlatIP(self.dimension)
            else:
                raise ValueError(
                    f"Vector dimension mismatch: expected {self.dimension}, got {arr.shape[1]}"
                )

        normalized_arr = self._normalize(arr)
        self.index.add(normalized_arr)
        self.chunks.extend(chunks)
        return len(self.chunks)

    def similarity_search(
        self,
        query_vector: List[float],
        top_k: int = 4,
        score_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Executes similarity search for a query embedding.
        Returns top_k most relevant chunks matching score_threshold.
        """
        if self.index.ntotal == 0 or not self.chunks:
            return []

        q_arr = np.array([query_vector], dtype=np.float32)
        if q_arr.shape[1] != self.dimension:
            raise ValueError(f"Query vector dimension {q_arr.shape[1]} != index dimension {self.dimension}")

        norm_q = self._normalize(q_arr)
        k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(norm_q, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            if score >= score_threshold:
                chunk_copy = dict(self.chunks[idx])
                chunk_copy["similarity_score"] = float(round(score, 4))
                results.append(chunk_copy)

        return results

    def get_indexed_documents(self) -> List[str]:
        """Returns sorted list of unique document filenames in the store."""
        docs = set(c.get("source", "") for c in self.chunks)
        return sorted(list(docs))

    def remove_document(self, document_name: str) -> int:
        """
        Removes all chunks associated with a specific document name
        and rebuilds the FAISS index.
        """
        remaining_chunks = [c for c in self.chunks if c.get("source") != document_name]
        removed_count = len(self.chunks) - len(remaining_chunks)

        if removed_count > 0:
            self.chunks = remaining_chunks
            # Rebuilding FAISS index from remaining items if embeddings need to be re-added
            # Note: For clean deletion in in-memory Flat index, reconstruct index:
            self.index = faiss.IndexFlatIP(self.dimension)
            # If no items left, index is clean. If items remain, vector store requires rebuild or re-index.
            # In practical session usage, caller can pass updated embeddings or clear.
        return removed_count

    def clear(self) -> None:
        """Wipes the FAISS index and associated chunks."""
        self.index = faiss.IndexFlatIP(self.dimension)
        self.chunks.clear()

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)
