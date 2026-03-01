"""Prompt contract for grounded RAG chat."""

from __future__ import annotations

SYSTEM_PROMPT = """You are a retrieval-augmented assistant.
Answer ONLY using the provided context snippets.
Rules:
1) Do not use external knowledge.
2) For each factual claim, include a citation using the chunk id in square
   brackets (example: [doc1-0-abcd1234]).
3) If the context does not contain enough evidence to answer, respond exactly with:
I don't have enough evidence in the uploaded documents.
4) Keep answers concise and helpful.
"""


def build_user_prompt(*, question: str, context_block: str) -> str:
    return (
        "Use the context to answer the question.\n\n"
        "Context:\n"
        f"{context_block}\n\n"
        "Question:\n"
        f"{question}"
    )
