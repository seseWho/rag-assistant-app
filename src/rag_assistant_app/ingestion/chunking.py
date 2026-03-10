"""Deterministic chunking helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from hashlib import sha1

logger = logging.getLogger(__name__)

# Separator hierarchy: try coarser splits first, fall back to finer ones.
_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", "; ", " "]


@dataclass(slots=True)
class Chunk:
    doc_id: str
    filename: str
    chunk_id: str
    text: str


def _make_chunk_id(doc_id: str, chunk_index: int, text: str) -> str:
    digest = sha1(f"{doc_id}:{chunk_index}:{text}".encode()).hexdigest()[:16]
    return f"{doc_id}-{chunk_index}-{digest}"


def _apply_separator(pieces: list[str], sep: str, chunk_size: int) -> list[str]:
    """Split any piece larger than chunk_size using sep."""
    result: list[str] = []
    for piece in pieces:
        if len(piece) <= chunk_size:
            result.append(piece)
        else:
            parts = [p.strip() for p in piece.split(sep) if p.strip()]
            result.extend(parts if parts else [piece])
    return result


def _hard_split(pieces: list[str], chunk_size: int) -> list[str]:
    """Character-level fallback for any piece still exceeding chunk_size."""
    result: list[str] = []
    for piece in pieces:
        if len(piece) <= chunk_size:
            result.append(piece)
        else:
            result.extend(piece[i : i + chunk_size] for i in range(0, len(piece), chunk_size))
    return [p for p in result if p]


def _split_into_atoms(text: str, chunk_size: int) -> list[str]:
    """Split text into pieces each <= chunk_size using natural separators."""
    pieces = [text]
    for sep in _SEPARATORS:
        if all(len(p) <= chunk_size for p in pieces):
            break
        pieces = _apply_separator(pieces, sep, chunk_size)
    return _hard_split(pieces, chunk_size)


def _next_chunk_start(atoms: list[str], start: int, end: int, chunk_overlap: int) -> int:
    """Walk back from end to find the atom index where the overlap window begins."""
    accumulated = 0
    new_start = end
    while new_start > start + 1:
        new_start -= 1
        accumulated += len(atoms[new_start]) + 1
        if accumulated >= chunk_overlap:
            break
    return new_start


def _merge_atoms(atoms: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    """Merge atoms into chunks of up to chunk_size chars with chunk_overlap char overlap."""
    if not atoms:
        return []

    chunks: list[str] = []
    start = 0

    while start < len(atoms):
        end = start
        current_len = 0
        while end < len(atoms):
            sep_len = 1 if end > start else 0
            if current_len + sep_len + len(atoms[end]) > chunk_size and end > start:
                break
            current_len += sep_len + len(atoms[end])
            end += 1

        if end == start:
            chunks.append(atoms[start])
            start += 1
            continue

        chunks.append("\n".join(atoms[start:end]))
        start = end if chunk_overlap == 0 else _next_chunk_start(atoms, start, end, chunk_overlap)

    return chunks


def chunk_text(
    *,
    doc_id: str,
    filename: str,
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """Split text into semantically-aware chunks with overlap.

    Splits along natural boundaries (paragraphs → sentences → words) before
    falling back to character splits, then merges small pieces up to chunk_size.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be in [0, chunk_size)")

    cleaned = text.strip()
    if not cleaned:
        return []

    atoms = _split_into_atoms(cleaned, chunk_size)
    merged = _merge_atoms(atoms, chunk_size, chunk_overlap)

    chunks: list[Chunk] = []
    for chunk_index, chunk_text_value in enumerate(merged):
        if not chunk_text_value.strip():
            continue
        chunk_id = _make_chunk_id(doc_id, chunk_index, chunk_text_value)
        chunks.append(
            Chunk(doc_id=doc_id, filename=filename, chunk_id=chunk_id, text=chunk_text_value)
        )

    logger.info(
        "chunk_text: doc=%s → %d chunk(s) (size=%d, overlap=%d)",
        doc_id,
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks
