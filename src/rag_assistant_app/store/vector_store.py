"""Simple persistent local vector store."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from rag_assistant_app.config import get_vector_store_dir
from rag_assistant_app.embeddings.embedder import Embedder


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

    def _persist(self) -> None:
        self.store_path.write_text(json.dumps(self._records, ensure_ascii=False), encoding="utf-8")

    def upsert_chunks(self, chunks: list[ChunkRecord]) -> None:
        embeddings = self.embedder.embed_documents([chunk.text for chunk in chunks])
        for chunk, embedding in zip(chunks, embeddings, strict=True):
            self._records[chunk.chunk_id] = {
                "chunk_id": chunk.chunk_id,
                "text": chunk.text,
                "metadata": chunk.metadata,
                "embedding": embedding,
            }
        self._persist()

    @staticmethod
    def _dot(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b, strict=True))


    def clear(self) -> None:
        self._records = {}
        self._persist()

    def query(self, text: str, top_k: int) -> list[RetrievedChunk]:
        if not self._records:
            return []
        query_embedding = self.embedder.embed_query(text)
        scored: list[RetrievedChunk] = []
        for item in self._records.values():
            score = self._dot(query_embedding, item["embedding"])
            scored.append(
                RetrievedChunk(
                    score=score,
                    chunk_id=item["chunk_id"],
                    text=item["text"],
                    metadata=item["metadata"],
                )
            )
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:top_k]
