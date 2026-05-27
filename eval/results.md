# RAG Ablation Study

## Experiment 1 — Chunk size: 300 vs 700 (k=5)

Compares how chunk granularity affects retrieval precision and answer quality at a fixed top-k of 5.

## Experiment 2 — Top-k: 3 vs 5 vs 8 (chunk_size=300, overlap=60)

Keeps the index identical and varies only how many chunks are passed to the LLM, trading recall against context noise.

## Results

| Experiment | chunk_size | overlap | k | collection_size | Hit@5 | MRR@5 | Answer accuracy | Notes |
|---|---|---|---|---|---|---|---|---|
| chunk300_overlap60 | 300 | 60 | 5 | 152 | 0.572 | 0.479 | 0.336 | Baseline — small chunks, k=5 |
| chunk700_overlap120 | 700 | 120 | 5 | 152 | 0.572 | 0.479 | 0.334 | Larger chunks — more context per chunk, k=5 |
| chunk300_overlap60_k3 | 300 | 60 | 3 | 152 | 0.524 | 0.468 | 0.376 | Fewer results — focused but may miss evidence |
| chunk300_overlap60_k8 | 300 | 60 | 8 | 152 | 0.655 | 0.491 | 0.334 | Broader retrieval window — more recall, more noise |

> **Answer accuracy** = mean token-level F1 between the generated answer and the reference answer, sampled over the first 20 grounded gold-set items.
> `N/A` means Ollama was unavailable during evaluation.
