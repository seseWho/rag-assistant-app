"""Logging helpers for rag_assistant_app."""

from __future__ import annotations

import logging

DEFAULT_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"



def configure_logging(level: int = logging.INFO) -> None:
    """Set a simple global logging configuration."""

    logging.basicConfig(level=level, format=DEFAULT_LOG_FORMAT)
