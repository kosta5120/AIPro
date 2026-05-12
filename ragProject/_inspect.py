"""Print document counts + a sample document for each source collection."""
from __future__ import annotations

import json
import sys

from pymongo import MongoClient

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from config import (
    MONGO_CHUNKS_COLLECTION,
    MONGO_COLLECTIONS,
    MONGO_DB,
    MONGO_URI,
)


def _truncate(s: str, n: int = 200) -> str:
    return s if len(s) <= n else s[:n] + "..."


def main() -> int:
    client = MongoClient(MONGO_URI)
    try:
        client.admin.command("ping")
        print(f"Connected to {MONGO_URI}, database '{MONGO_DB}'.")
        db = client[MONGO_DB]

        for name in MONGO_COLLECTIONS:
            coll = db[name]
            count = coll.estimated_document_count()
            print(f"\n=== {name} (count={count}) ===")
            sample = coll.find_one()
            if not sample:
                print("  (empty)")
                continue
            print("  fields:")
            for k, v in sample.items():
                preview = ""
                if isinstance(v, str):
                    preview = f" :: {_truncate(v)!r}"
                elif isinstance(v, (int, float, bool)):
                    preview = f" :: {v}"
                elif isinstance(v, list):
                    preview = f" :: list(len={len(v)})"
                elif isinstance(v, dict):
                    preview = f" :: dict(keys={list(v.keys())[:6]})"
                print(f"    {k:30s} {type(v).__name__}{preview}")
            print("  sample JSON (truncated):")
            print(
                "  "
                + _truncate(
                    json.dumps(sample, default=str, ensure_ascii=False),
                    1200,
                )
            )

        chunks_count = db[MONGO_CHUNKS_COLLECTION].estimated_document_count()
        print(
            f"\nvector store '{MONGO_CHUNKS_COLLECTION}': {chunks_count} chunks "
            "(run `python ingest.py` to populate)."
        )
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
