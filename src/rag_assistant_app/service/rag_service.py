"""Application service for indexing and retrieval."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import BinaryIO

logger = logging.getLogger(__name__)

from rag_assistant_app.embeddings.embedder import create_embedder
from rag_assistant_app.ingestion.chunking import chunk_text
from rag_assistant_app.ingestion.loaders import load_documents
from rag_assistant_app.ingestion.metadata import build_chunk_metadata
from rag_assistant_app.retrieval.retriever import Retriever
from rag_assistant_app.store.vector_store import ChunkRecord, LocalVectorStore, RetrievedChunk


@dataclass(slots=True)
class IndexSummary:
    docs_indexed: int
    chunks_indexed: int


class RagService:
    def __init__(self) -> None:
        embedder = create_embedder()
        self.vector_store = LocalVectorStore(embedder)
        self.retriever = Retriever(self.vector_store)

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
        chunk_records: list[ChunkRecord] = []
        for doc in documents:
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
        if chunk_records:
            self.vector_store.upsert_chunks(chunk_records)
        summary = IndexSummary(docs_indexed=len(documents), chunks_indexed=len(chunk_records))
        logger.info("index_documents: %s", summary)
        return summary

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        results = self.retriever.retrieve(query, top_k)
        logger.info(
            "retrieve: query=%r → %d chunk(s) returned (top score=%.4f)",
            query,
            len(results),
            results[0].score if results else float("nan"),
        )
        return results
