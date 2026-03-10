"""Gradio application entrypoint for local RAG assistant."""

from __future__ import annotations

import logging
from typing import Any

import gradio as gr

from rag_assistant_app.config import get_config
from rag_assistant_app.logging import configure_logging

# Configure logging before any service is instantiated so that startup
# messages (e.g. embedding model load) are captured.
configure_logging()

from rag_assistant_app.service.chat_service import ChatService  # noqa: E402

logger = logging.getLogger(__name__)

logger.info("Initialising ChatService…")
chat_service = ChatService()
logger.info("ChatService ready.")


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
        return "Please upload one or more `.txt`, `.md`, `.pdf` or `.docx` files first."

    logger.info("_index_documents: %d file(s) received, rebuild=%s", len(files), rebuild_index)
    # Gradio may pass NamedString (str subclass) or file-like objects.
    # Normalise to open binary handles using the path in all cases.
    paths = [f if isinstance(f, str) else f.name for f in files]
    logger.debug("File paths: %s", paths)

    with_handles = [open(p, "rb") for p in paths]
    try:
        summary = chat_service.rag_service.index_documents(
            with_handles,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            rebuild_index=rebuild_index,
        )
    except Exception:
        logger.exception("Indexing failed")
        return "❌ Indexing failed — check the console/logs for details."
    finally:
        for handle in with_handles:
            handle.close()

    return (
        "✅ Indexing complete. "
        f"Docs indexed: {summary.docs_indexed}. Chunks indexed: {summary.chunks_indexed}."
    )


def _history_to_pairs(history: list[dict]) -> list[tuple[str, str]]:
    """Convert Gradio messages format to (user, assistant) tuple pairs for the service layer."""
    pairs: list[tuple[str, str]] = []
    pending_user: str | None = None
    for msg in history:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "user":
            pending_user = content
        elif role == "assistant" and pending_user is not None:
            pairs.append((pending_user, content))
            pending_user = None
    return pairs


def _chat_turn(
    message: str,
    history: list[dict] | None,
    top_k: int,
    score_threshold: float,
):
    history = history or []
    logger.info("_chat_turn: question=%r, top_k=%d, threshold=%.4f", message, top_k, score_threshold)
    try:
        result = chat_service.answer(
            question=message,
            conversation_history=_history_to_pairs(history),
            top_k=top_k,
            score_threshold=score_threshold,
        )
    except Exception:
        logger.exception("_chat_turn: unexpected error")
        error_msg = (
            "❌ An unexpected error occurred — check the console/logs for details.\n\n"
            "**Tip:** if you see a dimension mismatch error, enable 'Rebuild index' and re-index your documents."
        )
        updated_history = history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": error_msg},
        ]
        return updated_history, updated_history, "Error during retrieval."

    updated_history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": result.answer},
    ]
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
                    label="Upload documents (.txt, .md, .pdf, .docx)",
                    file_count="multiple",
                    file_types=[".txt", ".md", ".pdf", ".docx"],
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
    app = build_app()
    app.launch()
