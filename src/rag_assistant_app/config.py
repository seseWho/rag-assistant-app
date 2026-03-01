"""Runtime configuration for the RAG assistant app."""

from __future__ import annotations

import os
from pathlib import Path


def get_vector_store_dir() -> Path:
    """Return the persistent vector store directory."""
    return Path(os.getenv("VECTOR_STORE_DIR", ".rag_store")).expanduser().resolve()


def get_embedding_model() -> str:
    """Return configured embedding model identifier."""
    return os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
