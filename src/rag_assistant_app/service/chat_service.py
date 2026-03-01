"""Chat orchestration service for RAG conversations."""

from __future__ import annotations

from dataclasses import dataclass

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

    def answer(
        self,
        *,
        question: str,
        conversation_history: list[tuple[str, str]],
        top_k: int,
        score_threshold: float,
        temperature: float = 0.0,
        max_tokens: int = 600,
    ) -> ChatResult:
        chunks = self.rag_service.retrieve(question, top_k=top_k)
        if not chunks:
            return ChatResult(answer=ABSTAIN_MESSAGE, retrieved_chunks=[])

        best_score = chunks[0].score
        if best_score < score_threshold:
            return ChatResult(answer=ABSTAIN_MESSAGE, retrieved_chunks=chunks)

        context_block = self._to_context_block(chunks)
        messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        for user_msg, assistant_msg in conversation_history:
            messages.append({"role": "user", "content": user_msg})
            messages.append({"role": "assistant", "content": assistant_msg})
        messages.append(
            {
                "role": "user",
                "content": build_user_prompt(question=question, context_block=context_block),
            }
        )

        try:
            answer = self.llm_client.chat_completion(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except LlmServiceError as exc:
            answer = f"⚠️ {exc}"

        return ChatResult(answer=answer or ABSTAIN_MESSAGE, retrieved_chunks=chunks)
