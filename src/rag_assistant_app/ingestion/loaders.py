"""Load supported user documents for indexing."""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".txt", ".md"}


@dataclass(slots=True)
class LoadedDocument:
    doc_id: str
    filename: str
    text: str


def _normalize_line_endings(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _read_text_from_bytes(raw: bytes, filename: str = "") -> str:
    for encoding in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            text = _normalize_line_endings(raw.decode(encoding))
            logger.debug("Decoded %s with encoding=%s", filename, encoding)
            return text
        except UnicodeDecodeError:
            continue
    logger.warning("All encodings failed for %s; using UTF-8 with replacement", filename)
    return _normalize_line_endings(raw.decode("utf-8", errors="replace"))


def load_documents(files: Iterable[BinaryIO]) -> list[LoadedDocument]:
    """Load uploaded txt/md files into memory.

    Expects file-like objects with a ``name`` attribute and ``read()`` method.
    """
    docs: list[LoadedDocument] = []
    for file in files:
        raw_name = getattr(file, "name", "unknown")
        filename = Path(raw_name).name
        suffix = Path(filename).suffix.lower()
        if suffix not in SUPPORTED_EXTENSIONS:
            logger.warning("Skipping unsupported file: %s (suffix=%s)", filename, suffix)
            continue

        logger.debug("Loading file: %s (raw path: %s)", filename, raw_name)
        raw = file.read()
        if isinstance(raw, str):
            text = _normalize_line_endings(raw)
        else:
            text = _read_text_from_bytes(raw, filename)
        logger.info("Loaded document: %s (%d chars)", filename, len(text))
        docs.append(LoadedDocument(doc_id=filename, filename=filename, text=text))
    logger.info("load_documents: %d document(s) loaded", len(docs))
    return docs
