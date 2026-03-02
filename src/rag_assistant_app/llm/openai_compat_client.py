"""OpenAI-compatible chat completion client."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import requests

from rag_assistant_app.config import get_config

logger = logging.getLogger(__name__)


class LlmServiceError(RuntimeError):
    """Raised when the configured LLM endpoint cannot be reached or returns invalid data."""


@dataclass(slots=True)
class OpenAICompatClient:
    base_url: str
    api_key: str
    model: str
    timeout_seconds: float = 60.0

    @classmethod
    def from_config(cls) -> OpenAICompatClient:
        config = get_config()
        return cls(base_url=config.llm_base_url, api_key=config.llm_api_key, model=config.llm_model)

    def chat_completion(
        self,
        *,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_tokens: int = 600,
    ) -> str:
        endpoint = f"{self.base_url.rstrip('/')}/chat/completions"
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

        logger.debug("POST %s  model=%s  messages=%d", endpoint, self.model, len(messages))
        try:
            response = requests.post(
                endpoint,
                headers=headers,
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            logger.error("LLM request failed: %s", exc, exc_info=True)
            raise LlmServiceError(
                "Could not reach the language model server. Please verify LM Studio is running."
            ) from exc

        try:
            content = str(data["choices"][0]["message"]["content"]).strip()
            logger.debug("LLM response: %d chars", len(content))
            return content
        except (KeyError, IndexError, TypeError) as exc:
            logger.error("Unexpected LLM response format: %s", data, exc_info=True)
            raise LlmServiceError("Language model response format was invalid.") from exc
