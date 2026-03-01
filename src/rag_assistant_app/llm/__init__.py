"""LLM client adapters."""

from .openai_compat_client import LlmServiceError, OpenAICompatClient

__all__ = ["LlmServiceError", "OpenAICompatClient"]
