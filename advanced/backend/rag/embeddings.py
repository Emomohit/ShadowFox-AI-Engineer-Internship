"""
Production Embedding Service for KnowledgeForge AI.
"""
from typing import List, Optional
import os
from dotenv import load_dotenv

try:
    from google import genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

from shared.mock_provider import MockEmbeddingProvider
from advanced.backend.core.exceptions import EmbeddingServiceError


class ProductionEmbeddingService:
    """Provides high-dimensional text embeddings for RAG indexing and search."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        use_mock: bool = False
    ):
        load_dotenv()
        self.api_key = api_key if api_key is not None else (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", "models/text-embedding-004")

        provider_env = os.getenv("LLM_PROVIDER", "").lower()
        if use_mock is True or provider_env == "mock":
            self.use_mock = True
        elif not self.api_key or self.api_key.strip() == "" or self.api_key == "your_gemini_api_key_here":
            if use_mock is False:
                raise EmbeddingServiceError("Valid Gemini API key required for live embeddings.")
            self.use_mock = True
        else:
            self.use_mock = False

        self._client = None
        self._mock = MockEmbeddingProvider(dimension=384)

        if not self.use_mock:
            if not GENAI_AVAILABLE:
                raise EmbeddingServiceError("google-genai library is not installed.")
            try:
                self._client = genai.Client(api_key=self.api_key)
            except Exception as e:
                raise EmbeddingServiceError(f"Failed to initialize Gemini Client: {str(e)}")

    def embed_query(self, query: str) -> List[float]:
        """Embeds a search query."""
        if not query or not query.strip():
            raise ValueError("Query string cannot be empty.")

        if self.use_mock:
            return self._mock.get_embedding(query)

        try:
            res = self._client.models.embed_content(
                model=self.model_name,
                contents=query
            )
            if hasattr(res, "embeddings") and res.embeddings:
                return res.embeddings[0].values
            elif hasattr(res, "embedding") and res.embedding:
                return res.embedding.values
            raise EmbeddingServiceError("Empty embedding returned by API.")
        except Exception as e:
            if isinstance(e, EmbeddingServiceError):
                raise e
            raise EmbeddingServiceError(f"Embedding API error: {str(e)}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Batch embeds chunks."""
        if not texts:
            return []

        if self.use_mock:
            return self._mock.get_embeddings(texts)

        return [self.embed_query(t) for t in texts]
