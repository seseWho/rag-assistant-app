"""ChromaDB-backed vector store."""

from __future__ import annotations

import logging
from pathlib import Path

import chromadb

from rag_assistant_app.config import get_vector_store_dir
from rag_assistant_app.embeddings.embedder import Embedder
from rag_assistant_app.store.vector_store import ChunkRecord, RetrievedChunk

logger = logging.getLogger(__name__)

_COLLECTION_NAME = "rag_documents"


class ChromaVectorStore:
    """Persistent vector store backed by ChromaDB with cosine similarity."""

    def __init__(self, embedder: Embedder, persist_dir: Path | None = None) -> None:
        self.embedder = embedder
        chroma_dir = (persist_dir or get_vector_store_dir()) / "chroma"
        chroma_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(chroma_dir))
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "ChromaVectorStore ready: %d record(s) at %s", self._collection.count(), chroma_dir
        )

    def upsert_chunks(self, chunks: list[ChunkRecord]) -> None:
        logger.info("Embedding %d chunk(s)…", len(chunks))
        embeddings = self.embedder.embed_documents([c.text for c in chunks])
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[c.metadata for c in chunks],
        )
        logger.info("Upserted %d chunk(s); store total: %d", len(chunks), self._collection.count())

    def list_documents(self) -> dict[str, int]:
        result = self._collection.get(include=["metadatas"])
        counts: dict[str, int] = {}
        for meta in result["metadatas"]:
            doc_id = meta.get("doc_id", "unknown")
            counts[doc_id] = counts.get(doc_id, 0) + 1
        return dict(sorted(counts.items()))

    def delete_document(self, doc_id: str) -> int:
        result = self._collection.get(where={"doc_id": doc_id}, include=[])
        ids = result["ids"]
        if ids:
            self._collection.delete(ids=ids)
        logger.info("delete_document: removed %d chunk(s) for doc_id=%s", len(ids), doc_id)
        return len(ids)

    def get_all_chunks(self) -> list[ChunkRecord]:
        result = self._collection.get(include=["documents", "metadatas"])
        docs = result["documents"] or []
        metas = result["metadatas"] or []
        return [
            ChunkRecord(chunk_id=cid, text=doc, metadata=meta)
            for cid, doc, meta in zip(result["ids"], docs, metas)
        ]

    def clear(self) -> None:
        self._client.delete_collection(_COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info("ChromaVectorStore cleared.")

    def query(self, text: str, top_k: int, doc_filter: set[str] | None = None) -> list[RetrievedChunk]:
        total = self._collection.count()
        if total == 0:
            logger.info("query: collection is empty, returning no results")
            return []

        query_embedding = self.embedder.embed_query(text)
        n_results = min(top_k, total)
        where = {"doc_id": {"$in": list(doc_filter)}} if doc_filter else None

        kwargs: dict = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        try:
            results = self._collection.query(**kwargs)
        except Exception:
            logger.exception("query: ChromaDB query failed")
            return []

        chunks: list[RetrievedChunk] = []
        for chunk_id, doc, meta, dist in zip(
            results["ids"][0],
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            # cosine distance = 1 - cosine_similarity → invert to get similarity score
            chunks.append(
                RetrievedChunk(score=1.0 - dist, chunk_id=chunk_id, text=doc, metadata=meta)
            )
        logger.debug("query: top-%d scores: %s", n_results, [f"{c.score:.4f}" for c in chunks])
        return chunks
