"""Load supported user documents for indexing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Iterable

SUPPORTED_EXTENSIONS = {".txt", ".md"}


@dataclass(slots=True)
class LoadedDocument:
    doc_id: str
    filename: str
    text: str


def _normalize_line_endings(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _read_text_from_bytes(raw: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return _normalize_line_endings(raw.decode(encoding))
        except UnicodeDecodeError:
            continue
    return _normalize_line_endings(raw.decode("utf-8", errors="replace"))


def load_documents(files: Iterable[BinaryIO]) -> list[LoadedDocument]:
    """Load uploaded txt/md files into memory.

    Expects file-like objects with a ``name`` attribute and ``read()`` method.
    """
    docs: list[LoadedDocument] = []
    for file in files:
        filename = Path(getattr(file, "name", "unknown")).name
        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            continue

        raw = file.read()
        if isinstance(raw, str):
            text = _normalize_line_endings(raw)
        else:
            text = _read_text_from_bytes(raw)
        docs.append(LoadedDocument(doc_id=filename, filename=filename, text=text))
    return docs
