"""Gradio application entrypoint for local RAG assistant."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import gradio as gr

from rag_assistant_app.config import get_chat_history_path, get_config
from rag_assistant_app.logging import configure_logging
from rag_assistant_app.prompts import SYSTEM_PROMPT

# Configure logging before any service is instantiated so that startup
# messages (e.g. embedding model load) are captured.
configure_logging()

from rag_assistant_app.service.chat_service import ChatService  # noqa: E402

logger = logging.getLogger(__name__)


def _load_history() -> list[dict]:
    path = get_chat_history_path()
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                logger.info("Chat history loaded: %d message(s) from %s", len(data), path)
                return data
        except Exception:
            logger.warning("Failed to load chat history from %s; starting fresh.", path, exc_info=True)
    return []


def _save_history(history: list[dict]) -> None:
    path = get_chat_history_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(history, ensure_ascii=False), encoding="utf-8")
        logger.debug("Chat history saved: %d message(s) to %s", len(history), path)
    except Exception:
        logger.warning("Failed to save chat history to %s.", path, exc_info=True)


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


def _doc_list_markdown() -> str:
    docs = chat_service.rag_service.list_documents()
    if not docs:
        return "_No documents indexed yet._"
    lines = ["| Document | Chunks |", "|---|---|"]
    for doc_id, count in docs.items():
        lines.append(f"| `{doc_id}` | {count} |")
    return "\n".join(lines)


_MAX_FILES = 20
_MAX_FILE_SIZE_MB = 50


def _doc_choices() -> list[str]:
    return list(chat_service.rag_service.list_documents().keys())


def _index_documents(
    files: list[Any] | None,
    chunk_size: int,
    chunk_overlap: int,
    rebuild_index: bool,
) -> tuple[str, str, Any]:
    if not files:
        msg = "Please upload one or more `.txt`, `.md`, `.pdf` or `.docx` files first."
        return msg, _doc_list_markdown(), gr.Dropdown(choices=_doc_choices())

    # --- pre-validation ---
    if len(files) > _MAX_FILES:
        msg = f"❌ Too many files: {len(files)} uploaded, maximum is {_MAX_FILES}."
        return msg, _doc_list_markdown(), gr.Dropdown(choices=_doc_choices())

    paths = [f if isinstance(f, str) else f.name for f in files]
    size_errors: list[str] = []
    for path in paths:
        size_mb = os.path.getsize(path) / (1024 * 1024)
        if size_mb > _MAX_FILE_SIZE_MB:
            size_errors.append(
                f"`{os.path.basename(path)}` ({size_mb:.1f} MB) exceeds the {_MAX_FILE_SIZE_MB} MB limit."
            )
    if size_errors:
        msg = "❌ File size limit exceeded:\n\n" + "\n".join(f"- {e}" for e in size_errors)
        return msg, _doc_list_markdown(), gr.Dropdown(choices=_doc_choices())

    logger.info("_index_documents: %d file(s) received, rebuild=%s", len(files), rebuild_index)
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
        msg = "❌ Indexing failed — check the console/logs for details."
        return msg, _doc_list_markdown(), gr.Dropdown(choices=_doc_choices())
    finally:
        for handle in with_handles:
            handle.close()

    msg = (
        "✅ Indexing complete. "
        f"Docs indexed: {summary.docs_indexed}. Chunks indexed: {summary.chunks_indexed}."
    )
    if summary.warnings:
        msg += "\n\n⚠️ Warnings:\n" + "\n".join(f"- {w}" for w in summary.warnings)
    return msg, _doc_list_markdown(), gr.Dropdown(choices=_doc_choices())


def _delete_document(doc_id: str) -> tuple[str, str, Any]:
    if not doc_id:
        return "Select a document to delete.", _doc_list_markdown(), gr.Dropdown(choices=_doc_choices())
    count = chat_service.rag_service.delete_document(doc_id)
    msg = f"✅ Deleted `{doc_id}` ({count} chunk(s) removed)."
    return msg, _doc_list_markdown(), gr.Dropdown(choices=_doc_choices(), value=None)


def _refresh_docs() -> tuple[str, Any]:
    return _doc_list_markdown(), gr.Dropdown(choices=_doc_choices())


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
    doc_filter: list[str] | None,
    system_prompt: str,
):
    history = list(history or [])
    active_filter: set[str] | None = set(doc_filter) if doc_filter else None
    logger.info(
        "_chat_turn: question=%r, top_k=%d, threshold=%.4f, doc_filter=%s",
        message, top_k, score_threshold, active_filter,
    )

    # Show the user message and a placeholder immediately
    history = history + [
        {"role": "user", "content": message},
        {"role": "assistant", "content": "▌"},
    ]
    yield history, history, "Retrieving…"

    result = None
    try:
        for result in chat_service.answer_stream(
            question=message,
            conversation_history=_history_to_pairs(history[:-2]),
            top_k=top_k,
            score_threshold=score_threshold,
            doc_filter=active_filter,
            system_prompt=system_prompt,
        ):
            history[-1]["content"] = result.answer + "▌"
            yield history, history, "Streaming…"
    except Exception:
        logger.exception("_chat_turn: unexpected error")
        error_msg = (
            "❌ An unexpected error occurred — check the console/logs for details.\n\n"
            "**Tip:** if you see a dimension mismatch error, enable 'Rebuild index' and re-index your documents."
        )
        history[-1]["content"] = error_msg
        yield history, history, "Error during retrieval."
        return

    if result is None:
        history[-1]["content"] = "❌ No response received."
        yield history, history, ""
        return

    history[-1]["content"] = result.answer
    _save_history(history)
    yield history, history, _format_retrieved_chunks(result.retrieved_chunks)


def build_app() -> gr.Blocks:
    """Create interactive chat + indexing UI."""

    config = get_config()

    with gr.Blocks(title="RAG Assistant App") as demo:
        gr.Markdown("# RAG Assistant App")
        gr.Markdown(
            f"LLM: `{config.llm_model}` at `{config.llm_base_url}`  \n"
            f"Embeddings: `{config.embedding_model}`"
        )

        history_state = gr.State(_load_history)

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

                with gr.Accordion("System prompt", open=False):
                    system_prompt_box = gr.Textbox(
                        label="System prompt",
                        value=SYSTEM_PROMPT,
                        lines=10,
                        show_label=False,
                    )
                    reset_prompt = gr.Button("Reset to default", scale=0)

                gr.Markdown("## Indexed Documents")
                doc_list = gr.Markdown(value=_doc_list_markdown)
                with gr.Row():
                    doc_selector = gr.Dropdown(
                        label="Select document",
                        choices=_doc_choices(),
                        value=None,
                        interactive=True,
                    )
                    refresh_button = gr.Button("↺", scale=0)
                delete_button = gr.Button("Delete selected", variant="stop")
                delete_status = gr.Markdown()

            with gr.Column(scale=2):
                gr.Markdown("## Chat")
                chatbot = gr.Chatbot(label="RAG chat", height=450)
                chat_doc_filter = gr.Dropdown(
                    label="Filter by document (empty = all docs)",
                    choices=_doc_choices(),
                    value=[],
                    multiselect=True,
                    interactive=True,
                )
                user_input = gr.Textbox(
                    label="Ask a question",
                    placeholder="What is in my uploaded docs?",
                )
                with gr.Row():
                    send = gr.Button("Send", variant="primary")
                    clear_history = gr.Button("Clear history", variant="stop", scale=0)
                with gr.Accordion("Retrieved chunks (last turn)", open=False):
                    retrieved_chunks = gr.Markdown("No turns yet.")

        def _index_and_refresh(files, chunk_size, chunk_overlap, rebuild_index):
            status, doc_md, selector = _index_documents(files, chunk_size, chunk_overlap, rebuild_index)
            return status, doc_md, selector, gr.Dropdown(choices=_doc_choices(), value=[])

        def _delete_and_refresh(doc_id):
            status, doc_md, selector = _delete_document(doc_id)
            return status, doc_md, selector, gr.Dropdown(choices=_doc_choices(), value=[])

        def _refresh_all():
            doc_md, selector = _refresh_docs()
            return doc_md, selector, gr.Dropdown(choices=_doc_choices())

        index_button.click(
            fn=_index_and_refresh,
            inputs=[uploads, chunk_size, chunk_overlap, rebuild_index],
            outputs=[index_status, doc_list, doc_selector, chat_doc_filter],
        )

        delete_button.click(
            fn=_delete_and_refresh,
            inputs=[doc_selector],
            outputs=[delete_status, doc_list, doc_selector, chat_doc_filter],
        )

        refresh_button.click(
            fn=_refresh_all,
            inputs=[],
            outputs=[doc_list, doc_selector, chat_doc_filter],
        )

        send.click(
            fn=_chat_turn,
            inputs=[user_input, history_state, top_k, score_threshold, chat_doc_filter, system_prompt_box],
            outputs=[chatbot, history_state, retrieved_chunks],
        )
        user_input.submit(
            fn=_chat_turn,
            inputs=[user_input, history_state, top_k, score_threshold, chat_doc_filter, system_prompt_box],
            outputs=[chatbot, history_state, retrieved_chunks],
        )

        def _clear_history():
            _save_history([])
            return [], []

        clear_history.click(fn=_clear_history, inputs=[], outputs=[chatbot, history_state])

        reset_prompt.click(fn=lambda: SYSTEM_PROMPT, inputs=[], outputs=[system_prompt_box])

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch()
