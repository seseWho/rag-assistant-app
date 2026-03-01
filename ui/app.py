"""Minimal Streamlit app for indexing and retrieval."""

from __future__ import annotations

import streamlit as st

from rag_assistant_app.service.rag_service import RagService

st.set_page_config(page_title="RAG Assistant", layout="wide")
st.title("RAG Assistant")

if "rag_service" not in st.session_state:
    st.session_state["rag_service"] = RagService()

service: RagService = st.session_state["rag_service"]

index_tab, retrieve_tab = st.tabs(["Index documents", "Retrieve"])

with index_tab:
    st.subheader("Upload text/markdown documents")
    files = st.file_uploader(
        "Choose .txt or .md files",
        type=["txt", "md"],
        accept_multiple_files=True,
    )
    if st.button("Index uploaded documents", type="primary"):
        summary = service.index_documents(files or [])
        st.success(
            f"Indexed {summary.docs_indexed} documents into {summary.chunks_indexed} chunks."
        )

with retrieve_tab:
    st.subheader("Retrieve top-k chunks")
    query = st.text_input("Query")
    top_k = st.number_input("Top-k", min_value=1, max_value=20, value=3)
    if st.button("Retrieve") and query.strip():
        results = service.retrieve(query=query.strip(), top_k=int(top_k))
        if not results:
            st.info("No chunks found. Index documents first.")
        for idx, chunk in enumerate(results, start=1):
            st.markdown(
                f"**{idx}. Score:** `{chunk.score:.4f}`  \\n"
                f"**Chunk ID:** `{chunk.chunk_id}`  \\n"
                f"**File:** `{chunk.metadata.get('filename', 'n/a')}`"
            )
            st.code(chunk.text)
