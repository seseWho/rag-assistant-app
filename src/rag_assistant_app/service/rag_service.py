"""Application service for indexing and retrieval."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import BinaryIO

logger = logging.getLogger(__name__)

from rag_assistant_app.config import get_config
from rag_assistant_app.embeddings.embedder import create_embedder
from rag_assistant_app.ingestion.chunking import chunk_text
from rag_assistant_app.ingestion.loaders import load_documents
from rag_assistant_app.ingestion.metadata import build_chunk_metadata
from rag_assistant_app.retrieval.hybrid_retriever import HybridRetriever
from rag_assistant_app.retrieval.reranker import RERANK_CANDIDATES_MULTIPLIER, create_reranker
from rag_assistant_app.store.vector_store import ChunkRecord, RetrievedChunk, VectorStore


_MIN_TEXT_CHARS = 20  # documents with less extractable text are skipped


@dataclass(slots=True)
class IndexSummary:
    docs_indexed: int
    chunks_indexed: int
    warnings: list[str]


def _create_vector_store(embedder) -> VectorStore:
    backend = get_config().vector_store_backend
    if backend == "local":
        from rag_assistant_app.store.vector_store import LocalVectorStore
        logger.info("Using LocalVectorStore backend (JSON file)")
        return LocalVectorStore(embedder)
    try:
        from rag_assistant_app.store.chroma_store import ChromaVectorStore
        logger.info("Using ChromaVectorStore backend")
        return ChromaVectorStore(embedder)
    except Exception as exc:
        from rag_assistant_app.store.vector_store import LocalVectorStore
        logger.warning(
            "ChromaVectorStore unavailable (%s); falling back to LocalVectorStore. "
            "Run `pip install --upgrade chromadb` to fix.",
            exc,
        )
        return LocalVectorStore(embedder)


class RagService:
    def __init__(self) -> None:
        embedder = create_embedder()
        self.vector_store = _create_vector_store(embedder)
        self.retriever = HybridRetriever(self.vector_store)
        self.reranker = create_reranker()

    def index_documents(
        self,
        files: list[BinaryIO],
        *,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        rebuild_index: bool = False,
    ) -> IndexSummary:
        if rebuild_index:
            self.vector_store.clear()

        documents = load_documents(files)
        warnings: list[str] = []
        chunk_records: list[ChunkRecord] = []
        docs_indexed = 0
        for doc in documents:
            if len(doc.text.strip()) < _MIN_TEXT_CHARS:
                msg = f"'{doc.filename}' produced no extractable text and was skipped."
                logger.warning("index_documents: %s", msg)
                warnings.append(msg)
                continue
            chunks = chunk_text(
                doc_id=doc.doc_id,
                filename=doc.filename,
                text=doc.text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            )
            for chunk in chunks:
                chunk_records.append(
                    ChunkRecord(
                        chunk_id=chunk.chunk_id,
                        text=chunk.text,
                        metadata=build_chunk_metadata(
                            doc_id=chunk.doc_id,
                            filename=chunk.filename,
                            chunk_id=chunk.chunk_id,
                        ),
                    )
                )
            docs_indexed += 1
        if chunk_records:
            self.vector_store.upsert_chunks(chunk_records)
        summary = IndexSummary(docs_indexed=docs_indexed, chunks_indexed=len(chunk_records), warnings=warnings)
        logger.info("index_documents: %s", summary)
        return summary

    def list_documents(self) -> dict[str, int]:
        """Return {doc_id: chunk_count} for all indexed documents."""
        return self.vector_store.list_documents()

    def delete_document(self, doc_id: str) -> int:
        """Delete all chunks for doc_id. Returns chunk count removed."""
        count = self.vector_store.delete_document(doc_id)
        logger.info("delete_document: doc_id=%s, removed=%d", doc_id, count)
        return count

    def retrieve(
        self, query: str, top_k: int = 3, doc_filter: set[str] | None = None, hybrid: bool = False
    ) -> list[RetrievedChunk]:
        fetch_k = top_k * RERANK_CANDIDATES_MULTIPLIER if self.reranker else top_k
        results = self.retriever.retrieve(query, fetch_k, doc_filter=doc_filter, hybrid=hybrid)

        if self.reranker and results:
            results = self.reranker.rerank(query, results, top_k)
            logger.info(
                "retrieve: reranked %d candidates → top-%d (top score=%.4f)",
                fetch_k,
                len(results),
                results[0].score if results else float("nan"),
            )
        else:
            logger.info(
                "retrieve: query=%r → %d chunk(s) returned (top score=%.4f)",
                query,
                len(results),
                results[0].score if results else float("nan"),
            )
        return results
