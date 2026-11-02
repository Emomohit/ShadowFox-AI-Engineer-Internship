"""
Text Chunker module for DocuMind AI.
Implements sliding-window recursive chunking with preserved metadata.
"""
from typing import List, Dict, Any


class TextChunker:
    """Splits document pages into semantically coherent chunks with preserved metadata."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        if chunk_size <= 50:
            raise ValueError("chunk_size must be greater than 50 characters.")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative.")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size.")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_pages(self, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Splits a list of page dicts into a list of chunk dicts:
        [{
            "chunk_id": str,
            "text": str,
            "source": str,
            "page_number": int,
            "chunk_index": int,
            "token_estimate": int
        }]
        """
        all_chunks = []
        global_chunk_idx = 0

        for page in pages:
            text = page.get("text", "")
            source = page.get("source", "unknown")
            page_num = page.get("page_number", 1)

            if not text:
                continue

            page_chunks = self._chunk_text(text)
            for idx, chunk_str in enumerate(page_chunks):
                chunk_id = f"{source}_p{page_num}_c{global_chunk_idx}"
                all_chunks.append({
                    "chunk_id": chunk_id,
                    "text": chunk_str,
                    "source": source,
                    "page_number": page_num,
                    "chunk_index": idx,
                    "global_index": global_chunk_idx,
                    "char_count": len(chunk_str),
                    "token_estimate": max(1, len(chunk_str) // 4)
                })
                global_chunk_idx += 1

        return all_chunks

    def _chunk_text(self, text: str) -> List[str]:
        """Splits a single block of text using sliding window with overlap."""
        chunks = []
        step = self.chunk_size - self.chunk_overlap
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + self.chunk_size, text_len)
            chunk = text[start:end].strip()

            if chunk:
                chunks.append(chunk)

            if end == text_len:
                break
            start += step

        return chunks
