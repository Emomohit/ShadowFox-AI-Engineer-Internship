"""
Document Loader module for DocuMind AI.
Extracts text and page-level metadata from PDF and TXT files with error handling.
"""
from typing import List, Dict, Any
import io
import os
import pypdf


class DocumentLoadingError(Exception):
    """Raised when document ingestion or parsing fails."""
    pass


class DocumentLoader:
    """Extracts text content and metadata from various file formats."""

    SUPPORTED_EXTENSIONS = {".pdf", ".txt"}

    @classmethod
    def load_file(cls, file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
        """
        Parses a file and returns a list of page/section dicts:
        [{"page_number": int, "text": str, "source": str, "char_count": int}]
        """
        if not file_bytes:
            raise DocumentLoadingError(f"File '{filename}' is completely empty (0 bytes).")

        ext = os.path.splitext(filename)[1].lower()
        if ext not in cls.SUPPORTED_EXTENSIONS:
            raise DocumentLoadingError(
                f"Unsupported file format '{ext}'. Only PDF and TXT are supported."
            )

        if ext == ".txt":
            return cls._load_txt(file_bytes, filename)
        elif ext == ".pdf":
            return cls._load_pdf(file_bytes, filename)
        else:
            raise DocumentLoadingError(f"Unsupported extension: {ext}")

    @classmethod
    def _load_txt(cls, file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
        try:
            # Attempt UTF-8 then fallback to Latin-1
            try:
                text = file_bytes.decode("utf-8")
            except UnicodeDecodeError:
                text = file_bytes.decode("latin-1")
            
            clean_text = text.strip()
            if not clean_text:
                raise DocumentLoadingError(f"Text file '{filename}' contains no readable text.")

            return [{
                "page_number": 1,
                "text": clean_text,
                "source": filename,
                "char_count": len(clean_text)
            }]
        except Exception as e:
            if isinstance(e, DocumentLoadingError):
                raise e
            raise DocumentLoadingError(f"Failed to decode TXT file '{filename}': {str(e)}")

    @classmethod
    def _load_pdf(cls, file_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
        try:
            stream = io.BytesIO(file_bytes)
            reader = pypdf.PdfReader(stream)
            num_pages = len(reader.pages)

            if num_pages == 0:
                raise DocumentLoadingError(f"PDF file '{filename}' contains 0 pages.")

            pages_data = []
            total_extracted_chars = 0

            for idx, page in enumerate(reader.pages):
                page_text = page.extract_text() or ""
                clean_page_text = page_text.strip()
                if clean_page_text:
                    pages_data.append({
                        "page_number": idx + 1,
                        "text": clean_page_text,
                        "source": filename,
                        "char_count": len(clean_page_text)
                    })
                    total_extracted_chars += len(clean_page_text)

            if total_extracted_chars == 0:
                raise DocumentLoadingError(
                    f"PDF '{filename}' appears to be a scanned image or empty. No selectable text could be extracted."
                )

            return pages_data
        except Exception as e:
            if isinstance(e, DocumentLoadingError):
                raise e
            raise DocumentLoadingError(f"Corrupt or unreadable PDF '{filename}': {str(e)}")
