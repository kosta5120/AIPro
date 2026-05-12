"""One-shot cleanup for the `rag_chunks` collection:
    1. Ensures the unique (source, source_id, chunk_index) + content_hash indexes
       exist.
    2. Backfills `content_hash` for chunks that don't have one yet
       (sha256 of UTF-8 bytes — same as ingest.py).
    3. Deletes duplicate chunks, keeping the one with the lowest `_id`
       per content hash.

Safe to run multiple times.
"""
from __future__ import annotations

import hashlib
import sys
#import pyodbc

#from config import CONN_STR
from bson.binary import Binary
from pymongo import ASCENDING, MongoClient
from pymongo import UpdateOne

from config import MONGO_CHUNKS_COLLECTION, MONGO_DB, MONGO_URI


def sha256_bytes(text: str) -> bytes:
    return hashlib.sha256((text or "").encode("utf-8")).digest()


def main() -> int:
    client = MongoClient(MONGO_URI)
    try:
        db = client[MONGO_DB]
        chunks = db[MONGO_CHUNKS_COLLECTION]

        print("Step 1/3: ensuring indexes...")
        chunks.create_index(
            [
                ("source", ASCENDING),
                ("source_id", ASCENDING),
                ("chunk_index", ASCENDING),
            ],
            name="uq_source_chunk",
            unique=True,
        )
        chunks.create_index([("content_hash", ASCENDING)], name="ix_content_hash")

        print("Step 2/3: backfilling content_hash for chunks missing it...")
        cur = chunks.find(
            {"$or": [{"content_hash": {"$exists": False}}, {"content_hash": None}]},
            {"_id": 1, "content": 1},
        )
        ops: list[UpdateOne] = []
        BATCH = 1000
        done = 0
        for doc in cur:
            h = sha256_bytes(doc.get("content") or "")
            ops.append(UpdateOne({"_id": doc["_id"]}, {"$set": {"content_hash": Binary(h)}}))
            if len(ops) >= BATCH:
                chunks.bulk_write(ops, ordered=False)
                done += len(ops)
                print(f"  updated {done}", end="\r", flush=True)
                ops = []
        if ops:
            chunks.bulk_write(ops, ordered=False)
            done += len(ops)
        print(f"  updated {done} chunks         ")

        print("Step 3/3: deleting duplicate chunks (keeping lowest _id per hash)...")
        pipeline = [
            {"$match": {"content_hash": {"$exists": True, "$ne": None}}},
            {
                "$group": {
                    "_id": "$content_hash",
                    "ids": {"$push": "$_id"},
                    "n": {"$sum": 1},
                }
            },
            {"$match": {"n": {"$gt": 1}}},
        ]
        dup_ids: list = []
        for group in chunks.aggregate(pipeline, allowDiskUse=True):
            keep = min(group["ids"])
            dup_ids.extend(_id for _id in group["ids"] if _id != keep)

        deleted = 0
        if dup_ids:
            CHUNK = 1000
            for i in range(0, len(dup_ids), CHUNK):
                res = chunks.delete_many({"_id": {"$in": dup_ids[i : i + CHUNK]}})
                deleted += res.deleted_count
        print(f"  deleted {deleted} duplicate chunks")

        remaining = chunks.estimated_document_count()
        print(f"\nDone. {remaining} unique chunks remain.")
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
