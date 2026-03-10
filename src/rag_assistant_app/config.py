"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class AppConfig:
    """Strongly named application configuration values."""

    llm_api_key: str
    llm_base_url: str
    llm_model: str
    llm_timeout_seconds: float
    embedding_model: str
    vector_store_dir: Path
    reranker_model: str  # empty string = disabled


def get_config() -> AppConfig:
    """Build configuration from environment variables."""

    return AppConfig(
        llm_api_key=os.getenv("LLM_API_KEY", "lm-studio"),
        llm_base_url=os.getenv("LLM_BASE_URL", "http://localhost:1234/v1"),
        llm_model=os.getenv("LLM_MODEL", "<your-lm-studio-model-id>"),
        llm_timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "300")),
        embedding_model=os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2"),
        vector_store_dir=Path(os.getenv("VECTOR_STORE_DIR", ".rag_store")),
        reranker_model=os.getenv("RERANKER_MODEL", ""),
    )


def get_llm_api_key() -> str:
    return get_config().llm_api_key


def get_llm_base_url() -> str:
    return get_config().llm_base_url


def get_llm_model() -> str:
    return get_config().llm_model


def get_embedding_model() -> str:
    return get_config().embedding_model


def get_vector_store_dir() -> Path:
    return get_config().vector_store_dir


def get_chat_history_path() -> Path:
    return get_vector_store_dir() / "chat_history.json"
