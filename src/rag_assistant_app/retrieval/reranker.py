"""Cross-encoder reranker for post-retrieval scoring."""

from __future__ import annotations

import logging

from rag_assistant_app.store.vector_store import RetrievedChunk

logger = logging.getLogger(__name__)

# Fetch this many times top_k from the vector store before reranking.
RERANK_CANDIDATES_MULTIPLIER = 3


class CrossEncoderReranker:
    """Reranks retrieved chunks using a cross-encoder model."""

    def __init__(self, model_name: str) -> None:
        from sentence_transformers import CrossEncoder

        logger.info("Loading CrossEncoder reranker: %s", model_name)
        self._model = CrossEncoder(model_name)
        self._model_name = model_name
        logger.info("CrossEncoder reranker loaded: %s", model_name)

    def rerank(self, query: str, chunks: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not chunks:
            return chunks
        pairs = [(query, chunk.text) for chunk in chunks]
        scores = self._model.predict(pairs)
        ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
        result = [chunk for _, chunk in ranked[:top_k]]
        logger.debug(
            "rerank: %d candidates → top-%d (model=%s)",
            len(chunks),
            len(result),
            self._model_name,
        )
        return result


def create_reranker() -> CrossEncoderReranker | None:
    from rag_assistant_app.config import get_config

    config = get_config()
    if not config.reranker_model:
        logger.info("Reranker disabled (RERANKER_MODEL not set)")
        return None
    try:
        reranker = CrossEncoderReranker(config.reranker_model)
        logger.info("Reranker ready (model=%s)", config.reranker_model)
        return reranker
    except Exception:
        logger.warning(
            "Failed to load reranker model '%s'; reranking disabled.",
            config.reranker_model,
            exc_info=True,
        )
        return None
