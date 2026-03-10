"""Embedding adapters: LM Studio API preferred, sentence-transformers fallback."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from hashlib import sha1, sha256
from math import sqrt
from pathlib import Path
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


class CachedEmbedder:
    """Transparent caching wrapper around any Embedder.

    Embeddings are persisted to a JSON file keyed by SHA-256 of the text.
    The cache is automatically invalidated when the model identifier changes.
    """

    def __init__(self, embedder: Embedder, model_id: str, cache_path: Path) -> None:
        self._embedder = embedder
        self._model_id = model_id
        self._cache_path = cache_path
        self._cache: dict[str, list[float]] = {}
        self._load()

    @staticmethod
    def _key(text: str) -> str:
        return sha256(text.encode("utf-8")).hexdigest()

    def _load(self) -> None:
        if not self._cache_path.exists():
            return
        try:
            data = json.loads(self._cache_path.read_text(encoding="utf-8"))
            if data.get("model") != self._model_id:
                logger.info(
                    "Embedding cache: model changed ('%s' → '%s'), invalidating cache.",
                    data.get("model"),
                    self._model_id,
                )
                return
            self._cache = data.get("embeddings", {})
            logger.info(
                "Embedding cache loaded: %d entry/ies from %s", len(self._cache), self._cache_path
            )
        except Exception:
            logger.warning("Failed to load embedding cache; starting fresh.", exc_info=True)

    def _save(self) -> None:
        try:
            self._cache_path.parent.mkdir(parents=True, exist_ok=True)
            self._cache_path.write_text(
                json.dumps(
                    {"model": self._model_id, "embeddings": self._cache}, ensure_ascii=False
                ),
                encoding="utf-8",
            )
            logger.debug("Embedding cache saved: %d entry/ies", len(self._cache))
        except Exception:
            logger.warning("Failed to save embedding cache.", exc_info=True)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        keys = [self._key(t) for t in texts]
        misses = [(i, texts[i]) for i, k in enumerate(keys) if k not in self._cache]

        if misses:
            logger.debug(
                "Embedding cache: %d hit(s), %d miss(es) — calling embedder.",
                len(texts) - len(misses),
                len(misses),
            )
            miss_indices, miss_texts = zip(*misses)
            new_embeddings = self._embedder.embed_documents(list(miss_texts))
            for idx, emb in zip(miss_indices, new_embeddings):
                self._cache[keys[idx]] = emb
            self._save()
        else:
            logger.debug("Embedding cache: all %d text(s) served from cache.", len(texts))

        return [self._cache[k] for k in keys]

    def embed_query(self, text: str) -> list[float]:
        key = self._key(text)
        if key not in self._cache:
            self._cache[key] = self._embedder.embed_query(text)
            self._save()
        return self._cache[key]


def create_embedder() -> Embedder:
    from rag_assistant_app.config import get_config, get_vector_store_dir

    config = get_config()
    cache_path = get_vector_store_dir() / "embedding_cache.json"

    # Try LM Studio embedding API first
    try:
        base = LMStudioEmbedder(config.llm_base_url, config.embedding_model, config.llm_api_key)
        base.embed_query("ping")  # verify the model is reachable
        logger.info("Using LMStudioEmbedder (model=%s)", config.embedding_model)
        return CachedEmbedder(base, config.embedding_model, cache_path)
    except Exception as exc:
        logger.warning(
            "LM Studio embedding not available (model=%s, url=%s): %s. "
            "Make sure the server is running (`lms server start`) and EMBEDDING_MODEL matches "
            "the identifier shown by `lms ps`. Falling back to HashingEmbedder.",
            config.embedding_model,
            config.llm_base_url,
            exc,
        )
        base_hashing = HashingEmbedder()
        return CachedEmbedder(base_hashing, f"hashing:{base_hashing.dim}", cache_path)
