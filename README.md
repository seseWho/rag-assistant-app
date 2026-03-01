# rag-assistant-app

A lightweight Gradio-based Retrieval-Augmented Generation (RAG) assistant scaffold.

This repository is **PR1 scaffolding only**: project layout, configuration, and a placeholder UI.
No functional RAG indexing/retrieval/generation pipeline is implemented yet.

## What this app is

- A local-first assistant app foundation.
- Chat + document upload UX will be built with Gradio.
- Planned integrations:
  - Local LLM via LM Studio's OpenAI-compatible API
  - Local embeddings (sentence-transformers)
  - Local vector store persistence (Chroma)

## Requirements

- Python 3.10+
- [LM Studio](https://lmstudio.ai/) running locally with a model loaded
- LM Studio local server enabled (default: `http://localhost:1234/v1`)

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .[dev]
```

## Configuration

1. Copy environment template:

```bash
cp .env.example .env
```

2. Update values in `.env` as needed (especially `LLM_MODEL`).

## Run

```bash
python -m ui.app
```

The app starts a minimal Gradio page and validates configuration loading.

## Project structure

```text
rag-assistant-app/
  README.md
  pyproject.toml
  .env.example
  .gitignore
  src/rag_assistant_app/
    __init__.py
    config.py
    logging.py
  ui/
    __init__.py
    app.py
  docs/
    architecture.md
```
