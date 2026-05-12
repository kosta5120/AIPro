"""Verify the MongoDB connection and create indexes on the rag_chunks
collection. Idempotent — safe to run repeatedly."""
from pymongo import ASCENDING, MongoClient

from config import (
    MONGO_CHUNKS_COLLECTION,
    MONGO_COLLECTIONS,
    MONGO_DB,
    MONGO_URI,
)


def main() -> int:
    client = MongoClient(MONGO_URI)
    try:
        client.admin.command("ping")
        db = client[MONGO_DB]
        chunks = db[MONGO_CHUNKS_COLLECTION]
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

        print(f"Connected to {MONGO_URI}, database '{MONGO_DB}'.")
        for name in MONGO_COLLECTIONS:
            count = db[name].estimated_document_count()
            print(f"  source collection {name}: {count} docs")
        print(
            f"  vector store collection '{MONGO_CHUNKS_COLLECTION}' is ready "
            f"({chunks.estimated_document_count()} chunks)."
        )
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
