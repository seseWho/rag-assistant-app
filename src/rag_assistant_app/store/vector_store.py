"""Simple persistent local vector store."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from rag_assistant_app.config import get_vector_store_dir
from rag_assistant_app.embeddings.embedder import Embedder

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ChunkRecord:
    chunk_id: str
    text: str
    metadata: dict[str, str]


@dataclass(slots=True)
class RetrievedChunk:
    score: float
    chunk_id: str
    text: str
    metadata: dict[str, str]


class LocalVectorStore:
    def __init__(self, embedder: Embedder, persist_dir: Path | None = None) -> None:
        self.embedder = embedder
        self.persist_dir = persist_dir or get_vector_store_dir()
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.store_path = self.persist_dir / "vectors.json"
        self._records: dict[str, dict] = {}
        self._load()

    def _load(self) -> None:
        if self.store_path.exists():
            self._records = json.loads(self.store_path.read_text(encoding="utf-8"))
            logger.info("Vector store loaded: %d record(s) from %s", len(self._records), self.store_path)
        else:
            logger.info("Vector store is empty (no file at %s)", self.store_path)

    def _persist(self) -> None:
        self.store_path.write_text(json.dumps(self._records, ensure_ascii=False), encoding="utf-8")
        logger.debug("Vector store persisted: %d record(s) to %s", len(self._records), self.store_path)

    def upsert_chunks(self, chunks: list[ChunkRecord]) -> None:
        logger.info("Embedding %d chunk(s)…", len(chunks))
        embeddings = self.embedder.embed_documents([chunk.text for chunk in chunks])
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            self._records[chunk.chunk_id] = {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "metadata": chunk.metadata,
                "embedding": embedding,
            }
        self._persist()
        logger.info("Upserted %d chunk(s); store total: %d", len(chunks), len(self._records))

    @staticmethod
    def _dot(a: list[float], b: list[float]) -> float:
        if len(a) != len(b):
            raise ValueError(
                f"Embedding dimension mismatch: query has {len(a)} dims but stored vector has {len(b)} dims. "
                "Clear the vector store (check 'Rebuild index') and re-index your documents."
            )
        return sum(x * y for x, y in zip(a, b))

    def clear(self) -> None:
        self._records = {}
        self._persist()
        logger.info("Vector store cleared.")

    def query(self, text: str, top_k: int) -> list[RetrievedChunk]:
        if not self._records:
            logger.info("query: vector store is empty, returning no results")
            return []
        query_embedding = self.embedder.embed_query(text)
        logger.debug("query: query_embedding dim=%d, searching %d records", len(query_embedding), len(self._records))
        scored: list[RetrievedChunk] = []
        for item in self._records.values():
            try:
                score = self._dot(query_embedding, item["embedding"])
            except ValueError:
                logger.error(
                    "Dimension mismatch for chunk '%s' (stored dim=%d, query dim=%d). "
                    "Re-index documents with 'Rebuild index' enabled.",
                    item["chunk_id"], len(item["embedding"]), len(query_embedding),
                    exc_info=True,
                )
                raise
            scored.append(
                RetrievedChunk(
                    score=score,
                    chunk_id=item["chunk_id"],
                    text=item["text"],
                    metadata=item["metadata"],
                )
            )
        scored.sort(key=lambda item: item.score, reverse=True)
        logger.debug("query: top-%d scores: %s", top_k, [f"{c.score:.4f}" for c in scored[:top_k]])
        return scored[:top_k]
