# RAG Assistant — Parameter Reference

## Indexing parameters

These parameters are set **before uploading documents** and control how the text is split into chunks. Changing them requires re-indexing.

---

### `chunk_size`

**Range:** 200 – 2000 · **Default:** 500

The maximum number of characters in each chunk. The document text is split into consecutive windows of this size.

| Value | Effect |
|---|---|
| Small (200–400) | More chunks, each with narrow context. Better precision for short, specific questions. |
| Medium (400–700) | Balanced. Good general-purpose default. |
| Large (700–2000) | Fewer chunks, each with broader context. Better for questions that need more surrounding information. |

**Rule of thumb:** use smaller chunks for factual Q&A over dense documents; larger chunks for summarisation or when answers span multiple sentences.

---

### `chunk_overlap`

**Range:** 0 – 400 · **Default:** 50

The number of characters shared between consecutive chunks. A chunk starting at position `N` and the next chunk starting at position `N + (chunk_size - chunk_overlap)` will share `chunk_overlap` characters.

```
chunk_size=500, chunk_overlap=50 → step=450
[0..500], [450..950], [900..1400], ...
```

| Value | Effect |
|---|---|
| 0 | No overlap. Sentences at chunk boundaries may be cut off mid-thought, losing context. |
| 50–100 | Slight overlap catches boundary sentences. Recommended default. |
| High (200+) | Heavy overlap, many redundant chunks. Slower indexing and retrieval; rarely needed. |

**Must be less than `chunk_size`.** A value of 0 is valid if documents are already well-structured.

---

## Retrieval parameters

These parameters are set **per query** in the chat panel and control what evidence is retrieved and whether the model answers at all.

---

### `top_k`

**Range:** 1 – 10 · **Default:** 3

The number of most-similar chunks retrieved from the vector store for each question. Only the top-k chunks are passed to the LLM as context.

| Value | Effect |
|---|---|
| 1–2 | Minimal context. Fast, but the answer may miss relevant information. |
| 3–5 | Balanced. Covers the main relevant passages without overloading the prompt. |
| 6–10 | Wide context. Useful for broad questions but increases prompt size and can dilute focus. |

Increasing `top_k` does **not** lower the score threshold — low-quality chunks can still be included. Use together with `score_threshold` to filter them out.

---

### `score_threshold`

**Range:** -1 – 1 · **Default:** 0.1

The minimum similarity score the **best** retrieved chunk must reach for the model to attempt an answer. Scores are dot-product similarity between unit-normalised vectors, so they range from -1 (opposite) to 1 (identical).

If the best chunk score is below this threshold, the app **abstains** and returns:
> *I don't have enough evidence in the uploaded documents.*

| Value | Effect |
|---|---|
| -1 | Never abstain. The model always answers, even with irrelevant context (risk of hallucination). |
| 0.0–0.1 | Loose threshold. Answers most questions; may include weakly related context. |
| 0.3–0.5 | Strict threshold. Only answers when strong evidence is found. Recommended for precision-critical use. |
| 1.0 | Effectively never answers (near-identical text required). |

**Tip:** start at `0.1` and raise it if the model gives answers unsupported by the documents.

---

## Parameter interaction example

| Goal | Suggested settings |
|---|---|
| Precise Q&A over a technical manual | `chunk_size=400`, `chunk_overlap=50`, `top_k=3`, `score_threshold=0.3` |
| Exploratory chat over long documents | `chunk_size=800`, `chunk_overlap=100`, `top_k=5`, `score_threshold=0.1` |
| Short documents, simple questions | `chunk_size=300`, `chunk_overlap=0`, `top_k=2`, `score_threshold=0.2` |
