"""Application service for indexing and retrieval."""

from __future__ import annotations

from dataclasses import dataclass
from typing import BinaryIO

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

    def index_documents(self, files: list[BinaryIO]) -> IndexSummary:
        documents = load_documents(files)
        chunk_records: list[ChunkRecord] = []
        for doc in documents:
            chunks = chunk_text(doc_id=doc.doc_id, filename=doc.filename, text=doc.text)
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
        return IndexSummary(docs_indexed=len(documents), chunks_indexed=len(chunk_records))

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        return self.retriever.retrieve(query, top_k)
