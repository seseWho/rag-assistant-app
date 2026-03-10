# RAG Assistant — Technical Architecture

## Overview

RAG Assistant is a **local-first** Retrieval-Augmented Generation application. All inference runs on-device via **LM Studio** (OpenAI-compatible API). No data leaves the machine. The application is structured in concentric layers, each with a single responsibility.

```text
┌─────────────────────────────────────────────┐
│                  UI (Gradio)                │  ui/app.py
├─────────────────────────────────────────────┤
│              Service layer                  │  chat_service / rag_service
├──────────────┬──────────────────────────────┤
│  Retrieval   │   Ingestion                  │  hybrid_retriever / loaders / chunking
├──────────────┴──────────────────────────────┤
│     Embedding   │   LLM client              │  embedder / openai_compat_client
├─────────────────┴───────────────────────────┤
│              Vector store                   │  ChromaVectorStore / LocalVectorStore
├─────────────────────────────────────────────┤
│     Persistence (.rag_store/ directory)     │
└─────────────────────────────────────────────┘
```

---

## Module map

```text
ui/
  app.py                          Gradio entrypoint, all UI event wiring

src/rag_assistant_app/
  config.py                       AppConfig dataclass, env-driven settings
  logging.py                      Structured logging setup

  service/
    chat_service.py               Orchestrates retrieval + LLM, streaming
    rag_service.py                Document indexing + retrieval facade

  ingestion/
    loaders.py                    Multi-format file loading (txt/md/pdf/docx)
    chunking.py                   Semantic chunking with overlap
    metadata.py                   Deterministic chunk ID generation

  embeddings/
    embedder.py                   LMStudioEmbedder, HashingEmbedder, CachedEmbedder

  retrieval/
    hybrid_retriever.py           BM25 + vector search fused with RRF
    reranker.py                   Optional CrossEncoder re-ranking

  store/
    vector_store.py               VectorStore Protocol + LocalVectorStore (JSON)
    chroma_store.py               ChromaVectorStore (ChromaDB, default)

  llm/
    openai_compat_client.py       HTTP client for LM Studio (blocking + streaming)

  prompts/
    chat_prompt.py                System prompt + user prompt builder
```

---

## Data flows

### 1. Document indexing

```text
User uploads files (UI)
        │
        ▼
Validation (file count ≤ 20, size ≤ 50 MB per file)
        │
        ▼
Loaders  ──  .txt / .md : multi-encoding (UTF-8 → UTF-8-sig → Latin-1)
          ├─ .pdf        : pymupdf, page text joined by \n\n
          └─ .docx       : python-docx, non-empty paragraphs joined by \n\n
        │
        ▼
Empty-text guard (< 20 chars → skip with warning)
        │
        ▼
SemanticChunker
  separator hierarchy: \n\n → \n → ". " → "? " → "! " → "; " → " "
  merge atoms up to chunk_size with chunk_overlap
  deterministic chunk ID: SHA1(doc_id:index:text)[:16]
        │
        ▼
CachedEmbedder.embed_documents()
  cache lookup by SHA-256(text)
  call LMStudioEmbedder only for misses
  persist cache to embedding_cache.json
        │
        ▼
VectorStore.upsert_chunks()   ← ChromaDB (HNSW, cosine) or LocalVectorStore (JSON)
```

### 2. Chat — retrieval phase

```text
User question (UI)
        │
        ▼
HybridRetriever.retrieve(query, top_k, doc_filter, hybrid)
  ├─ hybrid=False  →  VectorStore.query() only  (pure semantic search)
  └─ hybrid=True   →  BM25Okapi(full corpus) + VectorStore.query()
                       │                         │
                       │ keyword ranks            │ semantic ranks
                       └──────────┬──────────────┘
                                  │
                            RRF score = Σ 1 / (60 + rank_i)
                                  │
                            top-K by RRF score
        │
        ▼
CrossEncoderReranker.rerank()   (optional, if RERANKER_MODEL is set)
  fetch_k = top_k × 3 candidates → re-score with cross-encoder → top-K
        │
        ▼
Score threshold guard
  max(chunk.score) < score_threshold  →  abstain (no LLM call)
```

### 3. Chat — generation phase

```text
Retrieved chunks
        │
        ▼
_to_context_block()
  chunk_id / doc_id / score / snippet per chunk, separated by ---
        │
        ▼
_build_messages()
  [system]    custom or default system prompt
  [user/asst] conversation history pairs
  [user]      build_user_prompt(question, context_block)
        │
        ▼
OpenAICompatClient.chat_completion_stream()
  POST /v1/chat/completions  {stream: true}
  parse SSE lines → yield token strings
        │
        ▼
Gradio generator (yield ChatResult per token)
  UI updates chatbot in real time, shows ▌ cursor while streaming
        │
        ▼
On completion: _save_history() → chat_history.json
```

---

## Key design decisions

### Local-first embedding

`LMStudioEmbedder` calls the LM Studio `/v1/embeddings` endpoint instead of downloading models from HuggingFace. `HashingEmbedder` (deterministic bucket hashing) is the fallback when LM Studio is unavailable — ensuring the app always starts.

### Embedding cache

`CachedEmbedder` wraps any embedder with a SHA-256–keyed JSON cache persisted at `.rag_store/embedding_cache.json`. The cache stores the model name in its header and auto-invalidates when the model changes. Re-indexing unchanged documents costs zero API calls.

### Hybrid search with RRF

Reciprocal Rank Fusion merges keyword ranks (BM25) and semantic ranks (vector similarity) into a unified score without requiring tuned weights:

```text
RRF(chunk) = 1/(60 + rank_bm25) + 1/(60 + rank_vector)
```

BM25 helps with exact-match queries and rare technical terms; vector search handles paraphrasing and semantic similarity. The two are complementary.

### Abstention guardrail

`ChatService` skips the LLM call entirely and returns a fixed message when `max(retrieved_scores) < score_threshold`. This prevents hallucination when the document store has no relevant content. The check uses `max()` (not `chunks[0].score`) to be robust to reranking, which reorders chunks.

### Streaming

`chat_completion_stream()` opens an SSE connection to LM Studio and yields raw token strings. `_chat_turn()` in `app.py` is a Python generator; Gradio detects this and streams partial chatbot updates without any custom WebSocket code.

### Deterministic chunk IDs

```python
chunk_id = SHA1(f"{doc_id}:{index}:{text}")[:16]
```

IDs are stable across re-indexing runs. Upserting the same document twice is idempotent — existing chunks are overwritten, not duplicated.

### Vector store backends

| Backend | Storage | Search | When to use |
| --- | --- | --- | --- |
| `ChromaDB` (default) | SQLite + HNSW index in `.rag_store/chroma/` | Approximate nearest-neighbour | Any real workload |
| `LocalVectorStore` | Flat JSON in `.rag_store/vectors.json` | Exact dot-product scan | Testing / debugging |

Both implement the `VectorStore` Protocol so they are interchangeable. The backend is selected by `VECTOR_STORE_BACKEND` env var.

---

## Configuration reference

All settings are environment variables loaded from `.env` via `python-dotenv`.

| Variable | Default | Description |
| --- | --- | --- |
| `LLM_API_KEY` | `lm-studio` | API key for LM Studio (any string) |
| `LLM_BASE_URL` | `http://localhost:1234/v1` | LM Studio server URL |
| `LLM_MODEL` | _(required)_ | Model identifier as shown by `lms ps` |
| `LLM_TIMEOUT_SECONDS` | `300` | HTTP timeout for LLM requests |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Embedding model identifier (`lms ps`) |
| `VECTOR_STORE_DIR` | `.rag_store` | Root directory for all persistence |
| `RERANKER_MODEL` | _(empty)_ | HuggingFace cross-encoder model; leave empty to disable |
| `VECTOR_STORE_BACKEND` | `chroma` | `chroma` or `local` |
| `HYBRID_SEARCH` | `false` | Default value for the UI hybrid search checkbox |

---

## Persistence layout

```text
.rag_store/
├── chroma/               ChromaDB collection (SQLite + HNSW index files)
├── vectors.json          LocalVectorStore (used only when VECTOR_STORE_BACKEND=local)
├── embedding_cache.json  CachedEmbedder: {"model": "...", "embeddings": {"sha256": [...]}}
└── chat_history.json     Chat history: list of {"role": ..., "content": ...} dicts
```

---

## External dependencies

| Package | Role |
| --- | --- |
| `gradio` | Web UI framework |
| `chromadb` | Vector store with HNSW index |
| `rank-bm25` | BM25Okapi for keyword retrieval |
| `sentence-transformers` | Optional CrossEncoder reranker |
| `pymupdf` | PDF text extraction |
| `python-docx` | DOCX text extraction |
| `requests` | HTTP calls to LM Studio |
| `python-dotenv` | `.env` file loading |

---

## UI layout

```text
┌── Settings (left column, scale=1) ──────────────────────────────┐
│  Sliders: top_k · chunk_size · chunk_overlap · score_threshold   │
│  Checkbox: Hybrid search                                         │
│  Accordion: System prompt (editable) + Reset button             │
│                                                                  │
│  ## Upload & Index                                               │
│  File upload (.txt .md .pdf .docx) · Rebuild index checkbox     │
│  [Index documents]                                               │
│  Status message                                                  │
│                                                                  │
│  ## Indexed Documents                                            │
│  Markdown table (doc_id | chunks)                                │
│  Dropdown (select doc) · [↺ Refresh]                            │
│  [Delete selected]  · Status message                            │
└──────────────────────────────────────────────────────────────────┘

┌── Chat (right column, scale=2) ─────────────────────────────────┐
│  Chatbot (height 450)                                            │
│  Dropdown: Filter by document (multiselect, empty = all)        │
│  Textbox: Ask a question                                         │
│  [Send]  [Clear history]                                         │
│  Accordion: Retrieved chunks (last turn)                         │
└──────────────────────────────────────────────────────────────────┘
```
