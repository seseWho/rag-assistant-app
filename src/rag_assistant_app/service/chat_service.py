"""Chat orchestration service for RAG conversations."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass

logger = logging.getLogger(__name__)

from rag_assistant_app.llm import LlmServiceError, OpenAICompatClient
from rag_assistant_app.prompts import SYSTEM_PROMPT, build_user_prompt
from rag_assistant_app.service.rag_service import RagService
from rag_assistant_app.store.vector_store import RetrievedChunk

ABSTAIN_MESSAGE = "I don't have enough evidence in the uploaded documents."


@dataclass(slots=True)
class ChatResult:
    answer: str
    retrieved_chunks: list[RetrievedChunk]


class ChatService:
    def __init__(
        self,
        rag_service: RagService | None = None,
        llm_client: OpenAICompatClient | None = None,
    ):
        self.rag_service = rag_service or RagService()
        self.llm_client = llm_client or OpenAICompatClient.from_config()

    @staticmethod
    def _to_context_block(chunks: list[RetrievedChunk]) -> str:
        blocks: list[str] = []
        for chunk in chunks:
            doc_id = chunk.metadata.get("doc_id", "unknown")
            blocks.append(
                f"chunk_id: {chunk.chunk_id}\n"
                f"doc_id: {doc_id}\n"
                f"score: {chunk.score:.4f}\n"
                f"snippet: {chunk.text}"
            )
        return "\n\n---\n\n".join(blocks)

    def _build_messages(
        self,
        question: str,
        context_block: str,
        conversation_history: list[tuple[str, str]],
        system_prompt: str = SYSTEM_PROMPT,
    ) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        for user_msg, assistant_msg in conversation_history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})
        messages.append(
            {"role": "user", "content": build_user_prompt(question=question, context_block=context_block)}
        )
        return messages

    def _retrieve_and_check(
        self,
        question: str,
        top_k: int,
        score_threshold: float,
        doc_filter: set[str] | None = None,
        hybrid: bool = False,
    ) -> tuple[list[RetrievedChunk], bool]:
        """Return (chunks, should_abstain).

        Uses max embedding score across all chunks for the threshold check so
        that reranking (which reorders chunks) does not affect abstention logic.
        """
        chunks = self.rag_service.retrieve(question, top_k=top_k, doc_filter=doc_filter, hybrid=hybrid)
        if not chunks:
            logger.info("answer: no chunks retrieved → abstaining")
            return [], True
        best_score = max(c.score for c in chunks)
        if best_score < score_threshold:
            logger.info(
                "answer: best score %.4f < threshold %.4f → abstaining",
                best_score,
                score_threshold,
            )
            return chunks, True
        return chunks, False

    def answer(
        self,
        *,
        question: str,
        conversation_history: list[tuple[str, str]],
        top_k: int,
        score_threshold: float,
        doc_filter: set[str] | None = None,
        system_prompt: str = SYSTEM_PROMPT,
        hybrid: bool = False,
        temperature: float = 0.0,
        max_tokens: int = 600,
    ) -> ChatResult:
        chunks, should_abstain = self._retrieve_and_check(question, top_k, score_threshold, doc_filter=doc_filter, hybrid=hybrid)
        if should_abstain:
            return ChatResult(answer=ABSTAIN_MESSAGE, retrieved_chunks=chunks)

        messages = self._build_messages(question, self._to_context_block(chunks), conversation_history, system_prompt)
        logger.info("answer: sending %d message(s) to LLM (top_k=%d, threshold=%.4f)", len(messages), top_k, score_threshold)
        try:
            answer = self.llm_client.chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            logger.info("answer: LLM responded (%d chars)", len(answer) if answer else 0)
        except LlmServiceError as exc:
            logger.error("answer: LLM error: %s", exc)
            answer = f"⚠️ {exc}"

        return ChatResult(answer=answer or ABSTAIN_MESSAGE, retrieved_chunks=chunks)

    def answer_stream(
        self,
        *,
        question: str,
        conversation_history: list[tuple[str, str]],
        top_k: int,
        score_threshold: float,
        doc_filter: set[str] | None = None,
        system_prompt: str = SYSTEM_PROMPT,
        hybrid: bool = False,
        temperature: float = 0.0,
        max_tokens: int = 600,
    ) -> Iterator[ChatResult]:
        """Yield ChatResult objects with growing answer text as the LLM streams tokens."""
        chunks, should_abstain = self._retrieve_and_check(question, top_k, score_threshold, doc_filter=doc_filter, hybrid=hybrid)
        if should_abstain:
            yield ChatResult(answer=ABSTAIN_MESSAGE, retrieved_chunks=chunks)
            return

        messages = self._build_messages(question, self._to_context_block(chunks), conversation_history, system_prompt)
        logger.info("answer_stream: sending %d message(s) to LLM", len(messages))

        accumulated = ""
        try:
            for token in self.llm_client.chat_completion_stream(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            ):
                accumulated += token
                yield ChatResult(answer=accumulated, retrieved_chunks=chunks)
        except LlmServiceError as exc:
            logger.error("answer_stream: LLM error: %s", exc)
            yield ChatResult(answer=f"⚠️ {exc}", retrieved_chunks=chunks)
            return

        if not accumulated:
            yield ChatResult(answer=ABSTAIN_MESSAGE, retrieved_chunks=chunks)
