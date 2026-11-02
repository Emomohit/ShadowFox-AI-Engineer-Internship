"""
Embedding Service for DocuMind AI.
Encapsulates vector representation generation via Google GenAI or local fallback.
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


class EmbeddingError(Exception):
    """Raised when embedding generation fails."""
    pass


class EmbeddingService:
    """Generates dense vector embeddings for text chunks and queries."""

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
                raise EmbeddingError("Missing valid Gemini API key for embedding generation.")
            self.use_mock = True
        else:
            self.use_mock = False

        self._client = None
        self._mock_provider = MockEmbeddingProvider(dimension=384)

        if not self.use_mock:
            if not GENAI_AVAILABLE:
                raise EmbeddingError("google-genai SDK not installed.")
            try:
                self._client = genai.Client(api_key=self.api_key)
            except Exception as e:
                raise EmbeddingError(f"Failed to initialize Gemini Client for embeddings: {str(e)}")

    def embed_text(self, text: str) -> List[float]:
        """Embeds a single query or text string."""
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text.")

        if self.use_mock:
            return self._mock_provider.get_embedding(text)

        try:
            response = self._client.models.embed_content(
                model=self.model_name,
                contents=text
            )
            # Response contains embeddings list
            if hasattr(response, "embeddings") and response.embeddings:
                return response.embeddings[0].values
            elif hasattr(response, "embedding") and response.embedding:
                return response.embedding.values
            else:
                raise EmbeddingError("API returned empty embedding object.")
        except Exception as e:
            if isinstance(e, EmbeddingError):
                raise e
            raise EmbeddingError(f"Embedding API error: {str(e)}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embeds a batch of texts."""
        if not texts:
            return []

        if self.use_mock:
            return self._mock_provider.get_embeddings(texts)

        results = []
        for text in texts:
            results.append(self.embed_text(text))
        return results
