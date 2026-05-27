"""Retrieval interfaces.

Provides three retrievers, all exposing the same ``retrieve(query, k) -> list[dict]``
signature where each result has ``{chunk_id, text, score, metadata}``:

- :class:`Retriever`       — dense embedding retrieval via Chroma + SentenceTransformer
- :class:`BM25Retriever`   — sparse lexical retrieval via ``rank_bm25.BM25Okapi``
- :class:`HybridRetriever` — Reciprocal Rank Fusion of two underlying retrievers

The hybrid retriever fuses ranked results without needing score normalization,
which makes it robust when one retriever uses cosine similarity (0-1) and the
other uses raw BM25 scores (unbounded positives).
"""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from typing import Iterable

DEFAULT_PERSIST_DIR = os.path.join("data", "processed", "chroma")
DEFAULT_COLLECTION = "pdf_docs"
DEFAULT_CHUNKS_PATH = os.path.join("data", "processed", "chunks.jsonl")
EMBED_MODEL = "BAAI/bge-large-en-v1.5"

# Reciprocal Rank Fusion constant. 60 is the standard value from Cormack et al.
# (2009). It dampens the contribution of low-ranked items so that a doc ranked
# 5 and 7 is much better than a doc ranked 1 and 200.
RRF_K = 60


# ---------------------------------------------------------------------------
# Dense retriever (Chroma + SentenceTransformer)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=2)
def _load_model(name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name)


class Retriever:
    """Dense retriever backed by a persistent Chroma collection."""

    def __init__(
        self,
        persist_dir: str = DEFAULT_PERSIST_DIR,
        collection_name: str = DEFAULT_COLLECTION,
        embed_model: str = EMBED_MODEL,
    ):
        import chromadb

        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_collection(collection_name)
        self.model = _load_model(embed_model)

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        emb = self.model.encode(
            ["Represent this sentence for searching relevant passages: " + query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        res = self.collection.query(
            query_embeddings=emb.tolist(),
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        ids = res.get("ids", [[]])[0]
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        out: list[dict] = []
        for cid, text, md, dist in zip(ids, docs, metas, dists):
            score = 1.0 - float(dist) if dist is not None else 0.0
            out.append(
                {
                    "chunk_id": cid,
                    "text": text,
                    "score": score,
                    "metadata": md or {},
                }
            )
        return out

    def count(self) -> int:
        return self.collection.count()


# Alias for clarity in code that mixes retriever types.
DenseRetriever = Retriever


# ---------------------------------------------------------------------------
# BM25 retriever (rank_bm25, in-memory)
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    """Lower-cased, alphanumeric tokenisation. Keeps identifiers like ``BM25``
    and ``f1_score`` intact, which matters for a technical textbook."""
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def _read_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class BM25Retriever:
    """Sparse lexical retriever over the chunks JSONL produced by build_index.py.

    BM25 doesn't need a persisted vector store — it builds the index in-memory
    from the chunk corpus, which for this PDF takes <50 ms.
    """

    def __init__(self, chunks_path: str = DEFAULT_CHUNKS_PATH):
        from rank_bm25 import BM25Okapi

        if not os.path.exists(chunks_path):
            raise FileNotFoundError(
                f"BM25Retriever needs chunks at {chunks_path}. "
                f"Build the index first with `python src/build_index.py`."
            )
        self.chunks_path = chunks_path
        self.chunks: list[dict] = _read_jsonl(chunks_path)
        self.corpus_tokens: list[list[str]] = [
            _tokenize(c.get("text", "")) for c in self.chunks
        ]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []
        scores = self.bm25.get_scores(q_tokens)
        # argsort descending, take top-k
        import numpy as np

        top_idx = np.argsort(scores)[::-1][:k]
        out: list[dict] = []
        for i in top_idx:
            score = float(scores[i])
            if score <= 0.0:
                continue  # no lexical overlap — pointless to surface
            c = self.chunks[i]
            out.append(
                {
                    "chunk_id": c["chunk_id"],
                    "text":     c.get("text", ""),
                    "score":    score,
                    "metadata": c.get("metadata", {}) or {},
                }
            )
        return out

    def count(self) -> int:
        return len(self.chunks)


# ---------------------------------------------------------------------------
# Hybrid retriever (Reciprocal Rank Fusion)
# ---------------------------------------------------------------------------

class HybridRetriever:
    """Fuses results from two retrievers using Reciprocal Rank Fusion.

    For each retriever R we collect its top-N (N = ``pool_multiplier * k``)
    results. Each document gets a fused score of::

        rrf_score(d) = sum_R  1 / (RRF_K + rank_R(d))

    where ``rank_R(d)`` is the 1-based rank of ``d`` inside retriever ``R``'s
    top-N list. Documents appearing in only one list still get a score,
    documents appearing in both are boosted.

    RRF needs no score normalization, which is critical here because dense
    cosine similarity is in [0, 1] while BM25 scores are unbounded positives.
    """

    def __init__(
        self,
        dense: Retriever,
        sparse: BM25Retriever,
        pool_multiplier: int = 4,
        rrf_k: int = RRF_K,
    ):
        self.dense = dense
        self.sparse = sparse
        self.pool_multiplier = pool_multiplier
        self.rrf_k = rrf_k

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        pool = max(k * self.pool_multiplier, 10)
        dense_hits = self.dense.retrieve(query, k=pool)
        sparse_hits = self.sparse.retrieve(query, k=pool)

        # Aggregate by chunk_id
        fused: dict[str, dict] = {}
        for rank, hit in enumerate(dense_hits, start=1):
            cid = hit["chunk_id"]
            slot = fused.setdefault(cid, {**hit, "score": 0.0, "_sources": []})
            slot["score"] += 1.0 / (self.rrf_k + rank)
            slot["_sources"].append(("dense", rank))
        for rank, hit in enumerate(sparse_hits, start=1):
            cid = hit["chunk_id"]
            slot = fused.setdefault(cid, {**hit, "score": 0.0, "_sources": []})
            slot["score"] += 1.0 / (self.rrf_k + rank)
            slot["_sources"].append(("bm25", rank))

        ranked = sorted(fused.values(), key=lambda h: h["score"], reverse=True)[:k]
        # Strip the internal book-keeping field before returning.
        for r in ranked:
            r.pop("_sources", None)
        return ranked

    def count(self) -> int:
        return self.dense.count()
