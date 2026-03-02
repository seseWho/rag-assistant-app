"""Embedding adapters with sentence-transformers preferred."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from hashlib import sha1
from math import sqrt
from typing import Protocol

from rag_assistant_app.config import get_embedding_model

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


def create_embedder() -> Embedder:
    model_name = get_embedding_model()
    try:
        embedder = SentenceTransformerEmbedder(model_name)
        logger.info("Using SentenceTransformerEmbedder (model=%s, dim=inferred)", model_name)
        return embedder
    except Exception:
        logger.warning(
            "Failed to load SentenceTransformer model '%s'; falling back to HashingEmbedder.",
            model_name,
            exc_info=True,
        )
        return HashingEmbedder()
