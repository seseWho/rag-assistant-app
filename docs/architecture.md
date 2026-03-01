# Architecture (Scaffold)

## Current scope (PR1)

This repository currently provides only:

- Python project scaffolding and dependency management
- Environment-driven configuration for local LLM + embeddings + vector store settings
- Placeholder Gradio UI entrypoint
- Baseline logging helper

## Planned architecture (future PRs)

- **UI layer (Gradio):** chat interface, document upload, status messages
- **Application layer:** orchestration of ingestion, embedding, retrieval, and generation
- **Model clients:** LM Studio (OpenAI-compatible HTTP API)
- **Embedding layer:** local sentence-transformers model
- **Vector store:** local persisted index (Chroma directory)
- **Document pipeline:** file parsing, chunking, metadata handling

## Notes

No retrieval, indexing, or LLM inference flow is implemented in this PR.
