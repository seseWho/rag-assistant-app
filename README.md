# rag-assistant-app

A local-first Gradio Retrieval-Augmented Generation (RAG) assistant.

## Features (PR3)

- Upload `.txt` / `.md` documents and build a local index.
- Configure chunking (`chunk_size`, `chunk_overlap`) at index time.
- Ask questions in a chat UI backed by retrieval + LLM generation.
- Every response is expected to include chunk citations like `[<chunk_id>]`.
- Abstention guardrail when evidence is missing or weak:
  - `I don't have enough evidence in the uploaded documents.`
- Friendly error message if LM Studio is unavailable.

## Requirements

- Python 3.10+
- [LM Studio](https://lmstudio.ai/) running locally with a model loaded
- LM Studio local server enabled (OpenAI-compatible API)

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

2. Set environment values in `.env`:

```env
LLM_BASE_URL=http://localhost:1234/v1
LLM_API_KEY=lm-studio
LLM_MODEL=<your-lm-studio-model-id>
EMBEDDING_MODEL=all-MiniLM-L6-v2
VECTOR_STORE_DIR=.rag_store
```

## Run LM Studio

1. Open LM Studio.
2. Download/load a chat-capable model.
3. Start the local server (`Developer` tab).
4. Confirm base URL and model id match your `.env` values.

## Run the app

```bash
python -m ui.app
```

## How to use

### 1) Index documents

- In the sidebar, set:
  - `chunk_size`
  - `chunk_overlap`
  - optional `Rebuild index` to clear existing vectors
- Upload `.txt` or `.md` files.
- Click **Index documents**.

### 2) Chat over indexed data

- Set retrieval controls:
  - `top_k`: number of chunks to retrieve
  - `score_threshold`: minimum top score required to answer
- Ask questions in the chat panel.
- Expand **Retrieved chunks (last turn)** to inspect evidence used.

## Citations and abstention behavior

- The model is prompted to answer only from retrieved context.
- Responses should cite chunk ids in square brackets, for example `[mydoc.md-0-a1b2c3d4e5f6]`.
- If no chunks are retrieved, or retrieval confidence is below `score_threshold`, the app abstains with:
  - `I don't have enough evidence in the uploaded documents.`

## Notes

- If LM Studio is down or unreachable, the app returns a friendly warning in chat instead of crashing.
- The vector store persists locally at `VECTOR_STORE_DIR`.
