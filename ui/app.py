"""Placeholder Gradio app entrypoint."""

from __future__ import annotations

import gradio as gr

from rag_assistant_app.config import get_config
from rag_assistant_app.logging import configure_logging



def build_app() -> gr.Blocks:
    """Create a minimal placeholder UI while RAG features are under development."""

    config = get_config()

    with gr.Blocks(title="RAG Assistant App") as demo:
        gr.Markdown("# RAG Assistant App")
        gr.Markdown("🚧 Coming soon: chat + document upload + local RAG pipeline.")
        gr.Markdown(
            (
                "Configuration loaded successfully.\\n\\n"
                f"- LLM base URL: `{config.llm_base_url}`\\n"
                f"- LLM model: `{config.llm_model}`\\n"
                f"- Embedding model: `{config.embedding_model}`\\n"
                f"- Vector store dir: `{config.vector_store_dir}`"
            )
        )

    return demo


if __name__ == "__main__":
    configure_logging()
    app = build_app()
    app.launch()
