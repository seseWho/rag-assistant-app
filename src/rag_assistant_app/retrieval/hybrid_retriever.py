"""Hybrid retriever: BM25 + vector search fused with Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

import logging
import re

from rag_assistant_app.store.vector_store import RetrievedChunk, VectorStore

logger = logging.getLogger(__name__)

_RRF_K = 60  # standard constant; higher = less penalty for lower ranks


def _tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


class HybridRetriever:
    """Retrieves chunks using BM25 + vector similarity fused via RRF.

    When ``hybrid=False`` is passed to ``retrieve()``, it falls back to
    pure vector search (same behaviour as the plain Retriever).
    """

    def __init__(self, vector_store: VectorStore) -> None:
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        doc_filter: set[str] | None = None,
        hybrid: bool = True,
    ) -> list[RetrievedChunk]:
        if not hybrid:
            return self.vector_store.query(query, top_k, doc_filter=doc_filter)

        # ── 1. Fetch corpus ──────────────────────────────────────────────────
        all_chunks = self.vector_store.get_all_chunks()
        if doc_filter:
            all_chunks = [c for c in all_chunks if c.metadata.get("doc_id") in doc_filter]

        if not all_chunks:
            logger.info("hybrid_retrieve: no chunks in corpus, returning empty")
            return []

        # ── 2. BM25 rankings over the full (filtered) corpus ─────────────────
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.warning(
                "rank-bm25 not installed; falling back to pure vector search. "
                "Run `pip install rank-bm25` to enable hybrid search."
            )
            return self.vector_store.query(query, top_k, doc_filter=doc_filter)

        tokenized_corpus = [_tokenize(c.text) for c in all_chunks]
        bm25 = BM25Okapi(tokenized_corpus)
        bm25_scores = bm25.get_scores(_tokenize(query))

        # rank = position in descending-score order (1 = best)
        bm25_order = sorted(range(len(all_chunks)), key=lambda i: -bm25_scores[i])
        bm25_rank: dict[str, int] = {all_chunks[i].chunk_id: rank + 1 for rank, i in enumerate(bm25_order)}

        # ── 3. Vector search ─────────────────────────────────────────────────
        fetch_k = min(top_k * 3, len(all_chunks))
        vector_results = self.vector_store.query(query, fetch_k, doc_filter=doc_filter)
        vector_rank: dict[str, int] = {r.chunk_id: rank + 1 for rank, r in enumerate(vector_results)}
        vector_lookup: dict[str, RetrievedChunk] = {r.chunk_id: r for r in vector_results}

        # ── 4. RRF fusion ────────────────────────────────────────────────────
        candidates = set(bm25_rank) | set(vector_rank)
        rrf_scores: dict[str, float] = {}
        for cid in candidates:
            score = 0.0
            if cid in bm25_rank:
                score += 1.0 / (_RRF_K + bm25_rank[cid])
            if cid in vector_rank:
                score += 1.0 / (_RRF_K + vector_rank[cid])
            rrf_scores[cid] = score

        chunk_lookup: dict[str, RetrievedChunk] = {
            c.chunk_id: RetrievedChunk(score=0.0, chunk_id=c.chunk_id, text=c.text, metadata=c.metadata)
            for c in all_chunks
        }

        top_ids = sorted(rrf_scores, key=lambda cid: -rrf_scores[cid])[:top_k]
        results: list[RetrievedChunk] = []
        for cid in top_ids:
            base = vector_lookup.get(cid) or chunk_lookup[cid]
            results.append(
                RetrievedChunk(score=rrf_scores[cid], chunk_id=cid, text=base.text, metadata=base.metadata)
            )

        logger.info(
            "hybrid_retrieve: BM25+vector RRF → top-%d (scores: %s)",
            len(results),
            [f"{r.score:.4f}" for r in results],
        )
        return results
