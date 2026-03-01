"""Deterministic chunking helpers."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha1


@dataclass(slots=True)
class Chunk:
    doc_id: str
    filename: str
    chunk_id: str
    text: str


def _make_chunk_id(doc_id: str, chunk_index: int, text: str) -> str:
    digest = sha1(f"{doc_id}:{chunk_index}:{text}".encode()).hexdigest()[:16]
    return f"{doc_id}-{chunk_index}-{digest}"


def chunk_text(
    *,
    doc_id: str,
    filename: str,
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """Split text into fixed-size chunks with overlap."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be in [0, chunk_size)")

    cleaned = text.strip()
    if not cleaned:
        return []

    chunks: list[Chunk] = []
    step = chunk_size - chunk_overlap
    for chunk_index, start in enumerate(range(0, len(cleaned), step)):
        chunk_text_value = cleaned[start : start + chunk_size]
        if not chunk_text_value:
            continue
        chunk_id = _make_chunk_id(doc_id, chunk_index, chunk_text_value)
        chunks.append(
            Chunk(doc_id=doc_id, filename=filename, chunk_id=chunk_id, text=chunk_text_value)
        )
    return chunks
