"""Gradio application entrypoint for local RAG assistant."""

from __future__ import annotations

from typing import Any

import gradio as gr

from rag_assistant_app.config import get_config
from rag_assistant_app.logging import configure_logging
from rag_assistant_app.service.chat_service import ChatService

chat_service = ChatService()


def _format_retrieved_chunks(chunks: list[Any]) -> str:
    if not chunks:
        return "No chunks retrieved for the latest turn."

    lines = ["### Retrieved chunks"]
    for idx, chunk in enumerate(chunks, start=1):
        doc_id = chunk.metadata.get("doc_id", "unknown")
        snippet = chunk.text.replace("\n", " ").strip()
        if len(snippet) > 300:
            snippet = f"{snippet[:300]}..."
        lines.append(
            f"{idx}. `{chunk.chunk_id}` (doc: `{doc_id}`, score: `{chunk.score:.4f}`)\n"
            f"   - {snippet}"
        )
    return "\n".join(lines)


def _index_documents(
    files: list[Any] | None,
    chunk_size: int,
    chunk_overlap: int,
    rebuild_index: bool,
) -> str:
    if not files:
        return "Please upload one or more `.txt` or `.md` files first."

    with_handles = [open(file.name, "rb") for file in files]
    try:
        summary = chat_service.rag_service.index_documents(
            with_handles,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            rebuild_index=rebuild_index,
        )
    finally:
        for handle in with_handles:
            handle.close()

    return (
        "Indexing complete. "
        f"Docs indexed: {summary.docs_indexed}. Chunks indexed: {summary.chunks_indexed}."
    )


def _chat_turn(
    message: str,
    history: list[tuple[str, str]] | None,
    top_k: int,
    score_threshold: float,
):
    history = history or []
    result = chat_service.answer(
        question=message,
        conversation_history=history,
        top_k=top_k,
        score_threshold=score_threshold,
    )
    updated_history = history + [(message, result.answer)]
    return updated_history, updated_history, _format_retrieved_chunks(result.retrieved_chunks)


def build_app() -> gr.Blocks:
    """Create interactive chat + indexing UI."""

    config = get_config()

    with gr.Blocks(title="RAG Assistant App") as demo:
        gr.Markdown("# RAG Assistant App")
        gr.Markdown(
            f"LLM: `{config.llm_model}` at `{config.llm_base_url}`  \n"
            f"Embeddings: `{config.embedding_model}`"
        )

        history_state = gr.State([])

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("## Settings")
                top_k = gr.Slider(label="top_k", minimum=1, maximum=10, step=1, value=3)
                chunk_size = gr.Slider(
                    label="chunk_size", minimum=200, maximum=2000, step=50, value=500
                )
                chunk_overlap = gr.Slider(
                    label="chunk_overlap", minimum=0, maximum=400, step=10, value=50
                )
                score_threshold = gr.Slider(
                    label="score_threshold",
                    minimum=-1.0,
                    maximum=1.0,
                    step=0.01,
                    value=0.1,
                )

                gr.Markdown("## Upload & Index")
                uploads = gr.File(
                    label="Upload documents (.txt, .md)",
                    file_count="multiple",
                    file_types=[".txt", ".md"],
                )
                rebuild_index = gr.Checkbox(
                    label="Rebuild index (clear existing chunks first)", value=False
                )
                index_button = gr.Button("Index documents", variant="primary")
                index_status = gr.Markdown()

            with gr.Column(scale=2):
                gr.Markdown("## Chat")
                chatbot = gr.Chatbot(label="RAG chat", height=450)
                user_input = gr.Textbox(
                    label="Ask a question",
                    placeholder="What is in my uploaded docs?",
                )
                send = gr.Button("Send", variant="primary")
                with gr.Accordion("Retrieved chunks (last turn)", open=False):
                    retrieved_chunks = gr.Markdown("No turns yet.")

        index_button.click(
            fn=_index_documents,
            inputs=[uploads, chunk_size, chunk_overlap, rebuild_index],
            outputs=[index_status],
        )

        send.click(
            fn=_chat_turn,
            inputs=[user_input, history_state, top_k, score_threshold],
            outputs=[chatbot, history_state, retrieved_chunks],
        )
        user_input.submit(
            fn=_chat_turn,
            inputs=[user_input, history_state, top_k, score_threshold],
            outputs=[chatbot, history_state, retrieved_chunks],
        )

    return demo


if __name__ == "__main__":
    configure_logging()
    app = build_app()
    app.launch()
