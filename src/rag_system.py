"""RAG system entry point. Exposes `answer(question: str) -> dict`."""
from __future__ import annotations

import os
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from retrieval import Retriever, BM25Retriever, HybridRetriever  # noqa: E402
from generation import generate_answer, OllamaUnavailable  # noqa: E402

DEFAULT_TOP_K = int(os.environ.get("RAG_TOP_K", "5"))

_retriever: Optional[HybridRetriever] = None


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        dense = Retriever()
        sparse = BM25Retriever()
        _retriever = HybridRetriever(dense=dense, sparse=sparse)
    return _retriever


def answer(question: str, k: int = DEFAULT_TOP_K) -> dict:
    """Required interface: answer a question against the indexed corpus.

    Returns:
        {
            "answer": str,
            "sources": list[str],
            "retrieved_chunks": list[dict],
        }
    """
    retriever = get_retriever()
    retrieved = retriever.retrieve(question, k=k)

    if not retrieved:
        return {
            "answer": "The information was not found in the provided context.",
            "sources": [],
            "retrieved_chunks": [],
        }

    try:
        gen = generate_answer(question, retrieved)
        answer_text = gen["answer_text"]
        cited = gen["cited_ids"]
        if not cited:
            cited = [c["chunk_id"] for c in retrieved]
        return {
            "answer": answer_text,
            "sources": cited,
            "retrieved_chunks": retrieved,
        }
    except OllamaUnavailable as e:
        return {
            "answer": (
                "[generation unavailable] Retrieval succeeded but the LLM (Ollama) is "
                f"not reachable: {e}. Inspect retrieved_chunks for evidence."
            ),
            "sources": [c["chunk_id"] for c in retrieved],
            "retrieved_chunks": retrieved,
        }


if __name__ == "__main__":
    import json

    q = " ".join(sys.argv[1:]) or "What is the main topic of this document?"
    print(json.dumps(answer(q), ensure_ascii=False, indent=2))
