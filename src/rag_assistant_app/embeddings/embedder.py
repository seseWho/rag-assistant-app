"""Embedding adapters: LM Studio API preferred, sentence-transformers fallback."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from hashlib import sha1
from math import sqrt
from typing import Protocol

logger = logging.getLogger(__name__)


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...


@dataclass(slots=True)
class HashingEmbedder:
    """Deterministic fallback embedder using hashed token buckets."""

    dim: int = 384

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for token in text.lower().split():
            index = int(sha1(token.encode("utf-8")).hexdigest(), 16) % self.dim
            vector[index] += 1.0
        norm = sqrt(sum(v * v for v in vector)) or 1.0
        return [v / norm for v in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class LMStudioEmbedder:
    """Embedder that calls LM Studio's OpenAI-compatible /v1/embeddings endpoint."""

    def __init__(self, base_url: str, model: str, api_key: str = "lm-studio") -> None:
        import requests

        self._base_url = base_url.rstrip("/")
        self._model = model
        self._api_key = api_key
        self._requests = requests

    def _embed(self, texts: list[str]) -> list[list[float]]:
        endpoint = f"{self._base_url}/embeddings"
        headers = {"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"}
        response = self._requests.post(
            endpoint,
            json={"model": self._model, "input": texts},
            headers=headers,
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()
        vectors = [item["embedding"] for item in sorted(data["data"], key=lambda x: x["index"])]
        return [_normalize(v) for v in vectors]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        logger.debug("LMStudioEmbedder: embedding %d documents", len(texts))
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        logger.debug("LMStudioEmbedder: embedding query")
        return self._embed([text])[0]


class SentenceTransformerEmbedder:
    """Adapter around sentence-transformers."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import SentenceTransformer

        logger.info("Loading SentenceTransformer model: %s", model_name)
        self._model = SentenceTransformer(model_name)
        logger.info("SentenceTransformer model loaded: %s", model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True)
        return [vector.tolist() for vector in vectors]

    def embed_query(self, text: str) -> list[float]:
        vector = self._model.encode([text], normalize_embeddings=True)[0]
        return vector.tolist()


def _normalize(vector: list[float]) -> list[float]:
    norm = sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]


def create_embedder() -> Embedder:
    from rag_assistant_app.config import get_config

    config = get_config()

    # Try LM Studio embedding API first
    try:
        embedder = LMStudioEmbedder(config.llm_base_url, config.embedding_model, config.llm_api_key)
        embedder.embed_query("ping")  # verify the model is reachable
        logger.info("Using LMStudioEmbedder (model=%s)", config.embedding_model)
        return embedder
    except Exception as exc:
        logger.warning(
            "LM Studio embedding not available (model=%s, url=%s): %s. "
            "Make sure the server is running (`lms server start`) and EMBEDDING_MODEL matches "
            "the identifier shown by `lms ps`. Falling back to HashingEmbedder.",
            config.embedding_model,
            config.llm_base_url,
            exc,
        )
        return HashingEmbedder()
