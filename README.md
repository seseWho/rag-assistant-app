# rag-assistant-app

Minimal local RAG backend + Streamlit UI.

## Run

```bash
pip install -e .
streamlit run ui/app.py
```

Optional sentence-transformers support:

```bash
pip install -e '.[embeddings]'
```

## Environment variables

- `VECTOR_STORE_DIR` (default: `.rag_store`)
- `EMBEDDING_MODEL` (default: `sentence-transformers/all-MiniLM-L6-v2`)
