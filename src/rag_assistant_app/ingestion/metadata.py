"""Metadata helpers for indexed chunks."""

from __future__ import annotations


def build_chunk_metadata(*, doc_id: str, filename: str, chunk_id: str) -> dict[str, str]:
    return {"doc_id": doc_id, "filename": filename, "chunk_id": chunk_id}
