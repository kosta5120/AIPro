"""Pull posts from FacebookPosts / InstagramPosts / TwitterPosts (MongoDB),
chunk them, embed them, and store in the `rag_chunks` collection.

Re-runnable: skips chunks already present (source, source_id, chunk_index)
and chunks whose content sha256 is already in the index.
"""
from __future__ import annotations

import hashlib
import sys
from datetime import datetime, timezone
from typing import Any, Iterator

import numpy as np
from bson.binary import Binary
from pymongo import MongoClient, ASCENDING
from pymongo.collection import Collection
from pymongo.errors import BulkWriteError
from sentence_transformers import SentenceTransformer

from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    MONGO_CHUNKS_COLLECTION,
    MONGO_COLLECTIONS,
    MONGO_DB,
    MONGO_URI,
)


# Per-source-collection mapping: how to derive (platform tag, text body,
# author, natural id) from a raw document. The exact field names were
# discovered by inspecting the live MongoDB — see README.
SOURCE_MAP: dict[str, dict[str, Any]] = {
    "FacebookPosts": {
        "platform": "facebook",
        # FB posts split their body across UpperPost (main) and DownPost (trailing).
        "text_fields": ["UpperPost", "DownPost"],
        "author_fields": ["UserName", "UserUsername"],
        "date_field": "CreatedDate",
        "natural_id_field": "FacebookId",
    },
    "InstagramPosts": {
        "platform": "instagram",
        # Only `Text` carries the caption; the legacy SQL `Caption` column
        # does not exist in the Mongo documents.
        "text_fields": ["Text"],
        "author_fields": ["InstagramName", "InstagramUserName"],
        "date_field": "CreatedDate",
        "natural_id_field": "InstagramId",
    },
    "TwitterPosts": {
        "platform": "twitter",
        "text_fields": ["Text"],
        "author_fields": ["Name", "ScreenName"],
        "date_field": "CreatedDate",
        "natural_id_field": "TwitterId",
    },
}


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= size:
        return [text]
    step = size - overlap
    return [text[i : i + size] for i in range(0, len(text), step) if text[i : i + size].strip()]


def sha256_bytes(text: str) -> bytes:
    return hashlib.sha256((text or "").encode("utf-8")).digest()


def _first_nonempty(doc: dict, fields: list[str]) -> str:
    for f in fields:
        v = doc.get(f)
        if v is None:
            continue
        s = str(v).strip()
        if s:
            return s
    return ""


def _build_content(doc: dict, fields: list[str]) -> str:
    parts = []
    for f in fields:
        v = doc.get(f)
        if v is None:
            continue
        s = str(v).strip()
        if s:
            parts.append(s)
    return "\n".join(parts)


def ensure_chunks_indexes(coll: Collection) -> None:
    """Make repeated ingests cheap + safe."""
    coll.create_index(
        [("source", ASCENDING), ("source_id", ASCENDING), ("chunk_index", ASCENDING)],
        name="uq_source_chunk",
        unique=True,
    )
    coll.create_index([("content_hash", ASCENDING)], name="ix_content_hash")


def existing_keys(coll: Collection, platform: str) -> set[tuple[str, int]]:
    """(source_id, chunk_index) pairs already ingested for this platform."""
    cur = coll.find(
        {"source": platform},
        {"source_id": 1, "chunk_index": 1, "_id": 0},
    )
    return {(d["source_id"], d["chunk_index"]) for d in cur}


def existing_hashes(coll: Collection) -> set[bytes]:
    cur = coll.find(
        {"content_hash": {"$exists": True, "$ne": None}},
        {"content_hash": 1, "_id": 0},
    )
    out: set[bytes] = set()
    for d in cur:
        h = d.get("content_hash")
        if h is None:
            continue
        out.add(bytes(h))
    return out


def iter_new_chunks(
    source_coll: Collection,
    mapping: dict[str, Any],
    have: set[tuple[str, int]],
    seen_hashes: set[bytes],
) -> Iterator[dict]:
    """Yield ready-to-insert chunk documents (minus the embedding)."""
    platform = mapping["platform"]
    text_fields = mapping["text_fields"]
    author_fields = mapping["author_fields"]
    date_field = mapping["date_field"]
    natural_id_field = mapping["natural_id_field"]

    projection = {"_id": 1, date_field: 1, natural_id_field: 1}
    for f in text_fields + author_fields:
        projection[f] = 1

    for doc in source_coll.find({}, projection):
        src_id = str(doc.get("_id"))
        author = _first_nonempty(doc, author_fields)[:500]
        created = doc.get(date_field)
        if created is not None and not isinstance(created, datetime):
            try:
                created = datetime.fromisoformat(str(created))
            except Exception:
                created = None
        natural_id = doc.get(natural_id_field)
        if natural_id is not None:
            natural_id = str(natural_id)
        content = _build_content(doc, text_fields)
        for idx, piece in enumerate(chunk_text(content)):
            if (src_id, idx) in have:
                continue
            h = sha256_bytes(piece)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            yield {
                "source": platform,
                "source_collection": source_coll.name,
                "source_id": src_id,
                "natural_id": natural_id,
                "chunk_index": idx,
                "author": author or None,
                "created_date": created,
                "content": piece,
                "content_hash": Binary(h),
            }


def run_ingest(model: "SentenceTransformer | None" = None) -> int:
    """Embed-and-insert any new posts from MONGO_COLLECTIONS into rag_chunks.

    Importable from query.py so the embedding model can be reused. If `model`
    is None, this loads its own SentenceTransformer (preserves the standalone
    `python ingest.py` behaviour).
    """
    if model is None:
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        model = SentenceTransformer(EMBEDDING_MODEL)

    BATCH = 64
    grand_total = 0

    client = MongoClient(MONGO_URI)
    try:
        db = client[MONGO_DB]
        chunks = db[MONGO_CHUNKS_COLLECTION]
        ensure_chunks_indexes(chunks)
        seen_hashes = existing_hashes(chunks)
        print(f"Loaded {len(seen_hashes)} existing content hashes for dedup.")

        for source_name in MONGO_COLLECTIONS:
            mapping = SOURCE_MAP.get(source_name)
            if mapping is None:
                print(f"  (no mapping for {source_name}, skipping)")
                continue
            platform = mapping["platform"]
            print(f"\n--- {source_name} ({platform}) ---")
            source_coll = db[source_name]
            have = existing_keys(chunks, platform)

            buf: list[dict] = []
            count = 0

            def flush():
                nonlocal buf, count
                if not buf:
                    return
                texts = [d["content"] for d in buf]
                vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
                vecs = np.asarray(vecs, dtype=np.float32)
                now = datetime.now(timezone.utc)
                docs = []
                for d, v in zip(buf, vecs):
                    docs.append({**d, "embedding": Binary(v.tobytes()), "ingested_at": now})
                try:
                    chunks.insert_many(docs, ordered=False)
                except BulkWriteError as e:
                    # Unique-index races are fine — count only the successful writes.
                    inserted = e.details.get("nInserted", 0) if e.details else 0
                    count += inserted
                else:
                    count += len(docs)
                print(f"  inserted {count} chunks", end="\r", flush=True)
                buf = []

            for chunk_doc in iter_new_chunks(source_coll, mapping, have, seen_hashes):
                buf.append(chunk_doc)
                if len(buf) >= BATCH:
                    flush()
            flush()
            print(f"  inserted {count} chunks from {source_name}")
            grand_total += count
    finally:
        client.close()

    print(f"\nDone. {grand_total} new chunks ingested.")
    return 0


def main() -> int:
    return run_ingest()


if __name__ == "__main__":
    sys.exit(main())
