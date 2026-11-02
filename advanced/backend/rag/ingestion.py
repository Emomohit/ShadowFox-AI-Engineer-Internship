"""
Multi-Format Ingestion Pipeline for KnowledgeForge AI.
Supports PDF, TXT, and Markdown (MD) with stable hashing and typed metadata.
"""
from typing import List, Dict, Any, Tuple
import hashlib
import io
import os
import re
import pypdf
from advanced.backend.core.exceptions import IngestionError


class MultiFormatIngestor:
    """Validates, extracts, sanitizes, and chunks multi-format documents."""

    SUPPORTED_EXTS = {".pdf", ".txt", ".md"}

    @classmethod
    def process_file(
        cls,
        file_bytes: bytes,
        filename: str,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        max_size_bytes: int = 15 * 1024 * 1024
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        Validates, extracts, and chunks a file.
        Returns: (document_id, list_of_chunks)
        """
        if not file_bytes or len(file_bytes) == 0:
            raise IngestionError(f"Uploaded file '{filename}' is empty (0 bytes).")

        if len(file_bytes) > max_size_bytes:
            raise IngestionError(
                f"File '{filename}' ({len(file_bytes)} bytes) exceeds the limit of {max_size_bytes} bytes."
            )

        ext = os.path.splitext(filename)[1].lower()
        if ext not in cls.SUPPORTED_EXTS:
            raise IngestionError(
                f"Unsupported format '{ext}'. Supported formats are: {', '.join(cls.SUPPORTED_EXTS)}"
            )

        # Generate stable, reproducible document ID via SHA256
        doc_id = hashlib.sha256(file_bytes).hexdigest()[:16]

        # Extract pages
        if ext == ".pdf":
            pages = cls._extract_pdf(file_bytes, filename)
        elif ext in {".txt", ".md"}:
            pages = cls._extract_text_or_md(file_bytes, filename)
        else:
            raise IngestionError(f"Unsupported format: {ext}")

        # Chunk pages
        chunks = cls._chunk_pages(pages, doc_id, filename, chunk_size, chunk_overlap)
        if not chunks:
            raise IngestionError(f"No usable text segments could be derived from '{filename}'.")

        return doc_id, chunks

    @classmethod
    def _extract_text_or_md(cls, file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
        try:
            text = file_bytes.decode("utf-8")
        except UnicodeDecodeError:
            try:
                text = file_bytes.decode("latin-1")
            except Exception as e:
                raise IngestionError(f"Failed to decode text file '{filename}': {str(e)}")

        sanitized = cls._sanitize_text(text)
        if not sanitized:
            raise IngestionError(f"Text file '{filename}' contains only whitespace or unprintable characters.")

        return [{
            "page_number": 1,
            "text": sanitized
        }]

    @classmethod
    def _extract_pdf(cls, file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
        try:
            stream = io.BytesIO(file_bytes)
            reader = pypdf.PdfReader(stream)
            num_pages = len(reader.pages)

            if num_pages == 0:
                raise IngestionError(f"PDF file '{filename}' has 0 pages.")

            pages = []
            for idx, p in enumerate(reader.pages):
                raw = p.extract_text() or ""
                clean = cls._sanitize_text(raw)
                if clean:
                    pages.append({
                        "page_number": idx + 1,
                        "text": clean
                    })

            if not pages:
                raise IngestionError(f"PDF '{filename}' contains no selectable text (scanned image or empty).")

            return pages
        except Exception as e:
            if isinstance(e, IngestionError):
                raise e
            raise IngestionError(f"Corrupt PDF structure in '{filename}': {str(e)}")

    @classmethod
    def _sanitize_text(cls, text: str) -> str:
        """Strips non-printable chars, normalizes unicode spaces and consecutive newlines."""
        if not text:
            return ""
        # Remove null bytes and control chars except newlines and tabs
        text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)
        # Normalize multiple blank lines to double newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    @classmethod
    def _chunk_pages(
        cls,
        pages: List[Dict[str, Any]],
        doc_id: str,
        filename: str,
        chunk_size: int,
        chunk_overlap: int
    ) -> List[Dict[str, Any]]:
        """Splits pages into overlapping chunks with stable IDs."""
        chunks = []
        step = max(1, chunk_size - chunk_overlap)
        global_idx = 0

        for page in pages:
            page_num = page["page_number"]
            text = page["text"]
            text_len = len(text)
            start = 0

            while start < text_len:
                end = min(start + chunk_size, text_len)
                segment = text[start:end].strip()

                if segment:
                    chunk_id = f"{doc_id}_p{page_num}_c{global_idx}"
                    chunks.append({
                        "chunk_id": chunk_id,
                        "document_id": doc_id,
                        "document_name": filename,
                        "page_number": page_num,
                        "chunk_index": global_idx,
                        "text": segment,
                        "char_count": len(segment)
                    })
                    global_idx += 1

                if end == text_len:
                    break
                start += step

        return chunks
