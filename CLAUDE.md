# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install (editable mode with dev deps)
pip install -e .[dev]

# Run the app (Gradio UI at http://localhost:7860)
python -m ui.app

# Tests
pytest

# Lint / format
ruff check .
ruff format .

# Type checking
mypy src
```

**Runtime requirement:** LM Studio must be running locally with a loaded model and OpenAI-compatible server enabled (default: `http://localhost:1234/v1`). Copy `.env.example` to `.env` and set `LLM_MODEL` to your loaded model ID.

## Architecture

Layered: **UI → Services → Retrieval/LLM → Storage**

```
ui/app.py                    # Gradio entrypoint, calls build_app()
src/rag_assistant_app/
├── service/
│   ├── chat_service.py      # Orchestrates RAG + abstention + LLM
│   └── rag_service.py       # Indexes documents, wraps retrieval
├── retrieval/retriever.py   # Top-k similarity search over vector store
├── store/vector_store.py    # JSON-persisted LocalVectorStore (dot-product similarity)
├── embeddings/embedder.py   # SentenceTransformerEmbedder with HashingEmbedder fallback
├── ingestion/               # loaders.py (multi-encoding), chunking.py (fixed-size + overlap), metadata.py
├── llm/openai_compat_client.py  # HTTP client for LM Studio OpenAI-compatible API
├── prompts/chat_prompt.py   # System + user prompt builders
└── config.py                # AppConfig dataclass, env-driven
```

**Document indexing flow:** File upload → `Loaders` (multi-encoding) → `Chunking` (fixed-size with overlap, deterministic SHA1 IDs) → `Embedder` → `LocalVectorStore` (persisted to `.rag_store/vectors.json`)

**Chat flow:** User query → `RagService.retrieve()` → if below score threshold, return abstention → format context block → `OpenAICompatClient` sends system prompt + context + history → response with `[chunk_id]` citations

**Abstention:** `ChatService` abstains (returns a message without calling the LLM) if retrieval yields no results or all scores fall below the configured threshold. This prevents hallucination when the document store has no relevant content.

## Key Implementation Details

- **Chunk IDs** are deterministic: `SHA1(doc_id:index:text)[:16]`, enabling idempotent re-indexing.
- **Embeddings** normalize to unit vectors (L2); similarity uses dot product.
- **ChromaDB** is installed as a dependency but is not used; the custom `LocalVectorStore` is used instead.
- **Configuration** is entirely env-driven via `AppConfig` (`config.py`); no hardcoded defaults except fallback values in the dataclass.
- **Ruff** is configured with rules E, F, I, UP and line length 100. Run `ruff check .` before committing.
