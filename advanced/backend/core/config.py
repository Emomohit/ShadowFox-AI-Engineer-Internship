"""
KnowledgeForge AI - Core Settings & Configuration.
Pydantic V2 BaseSettings configuration.
"""
import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    """Application settings with environment variable bindings."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    APP_NAME: str = "KnowledgeForge AI - Production RAG Backend"
    APP_VERSION: str = "1.0.0"
    APP_ENV: str = "development"

    # LLM & Embedding Settings
    LLM_PROVIDER: str = "gemini"
    GEMINI_API_KEY: str = ""
    LLM_MODEL: str = "gemini-1.5-flash"
    EMBEDDING_MODEL: str = "models/text-embedding-004"

    # Ingestion & Chunking Defaults
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 50
    MAX_FILE_SIZE_BYTES: int = 15 * 1024 * 1024  # 15 MB limit

    # Retrieval Defaults
    DEFAULT_TOP_K: int = 4
    SIMILARITY_THRESHOLD: float = 0.05
    RERANKING_ENABLED: bool = True

    # Directories
    BASE_DIR: str = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    STORAGE_DIR: str = os.path.join(BASE_DIR, "storage")


settings = Settings()
