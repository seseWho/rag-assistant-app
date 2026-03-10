"""Retriever abstraction over vector store."""

from __future__ import annotations

from rag_assistant_app.store.vector_store import LocalVectorStore, RetrievedChunk


class Retriever:
    def __init__(self, vector_store: LocalVectorStore) -> None:
        self.vector_store = vector_store

    def retrieve(self, query: str, top_k: int = 3, doc_filter: set[str] | None = None) -> list[RetrievedChunk]:
        return self.vector_store.query(query, top_k, doc_filter=doc_filter)
