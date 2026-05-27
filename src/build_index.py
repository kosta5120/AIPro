"""Build a Chroma vector index over a PDF corpus.

Usage:
    python src/build_index.py --chunk-strategy fixed --chunk-size 500 --overlap 80
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

# Allow running as a script from the repo root.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import load_all_docs, chunk_documents, write_jsonl  # noqa: E402

DEFAULT_PERSIST_DIR = os.path.join("data", "processed", "chroma")
DEFAULT_CHUNKS_PATH = os.path.join("data", "processed", "chunks.jsonl")
DEFAULT_COLLECTION = "pdf_docs"
EMBED_MODEL = "BAAI/bge-large-en-v1.5"


def parse_args():
    p = argparse.ArgumentParser(description="Build the Chroma vector index.")
    p.add_argument("--raw-dir", default="data/raw")
    p.add_argument(
        "--load-mode",
        choices=["full", "page", "structured"],
        default="structured",
        help=(
            "How to read each PDF: 'full' = one doc per file, 'page' = one doc "
            "per page, 'structured' = one doc per Module/Concept/Subsection "
            "(headings detected by font size; chunks inherit section metadata)."
        ),
    )
    p.add_argument("--chunk-strategy", choices=["fixed", "sentence", "recursive"], default="recursive")
    p.add_argument("--chunk-size", type=int, default=1500)
    p.add_argument("--overlap", type=int, default=200)
    p.add_argument("--max-per-source", type=int, default=1500)
    p.add_argument("--collection-name", default=DEFAULT_COLLECTION)
    p.add_argument("--persist-dir", default=DEFAULT_PERSIST_DIR)
    p.add_argument("--chunks-path", default=DEFAULT_CHUNKS_PATH)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument(
        "--clean",
        action="store_true",
        help="Delete the persist dir before building (useful between runs).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    import chromadb
    from sentence_transformers import SentenceTransformer
    from tqdm import tqdm

    print(
        f"[1/5] Loading docs from {args.raw_dir} "
        f"(mode={args.load_mode}, max {args.max_per_source} per source)..."
    )
    docs = load_all_docs(
        args.raw_dir,
        max_per_source=args.max_per_source,
        mode=args.load_mode,
    )
    print(f"  loaded {len(docs)} docs")

    print(
        f"[2/5] Chunking strategy={args.chunk_strategy} size={args.chunk_size} overlap={args.overlap}..."
    )
    chunks = chunk_documents(
        docs,
        strategy=args.chunk_strategy,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
    print(f"  produced {len(chunks)} chunks")
    if not chunks:
        print("ERROR: no chunks produced. Aborting.")
        sys.exit(1)

    print(f"[3/5] Writing chunks to {args.chunks_path} ...")
    write_jsonl(args.chunks_path, chunks)

    if args.clean and os.path.isdir(args.persist_dir):
        print(f"  cleaning {args.persist_dir}")
        shutil.rmtree(args.persist_dir)
    os.makedirs(args.persist_dir, exist_ok=True)

    print(f"[4/5] Loading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)

    print(f"[5/5] Building Chroma collection '{args.collection_name}' at {args.persist_dir}")
    client = chromadb.PersistentClient(path=args.persist_dir)
    try:
        client.delete_collection(args.collection_name)
    except Exception:
        pass
    collection = client.create_collection(
        name=args.collection_name,
        metadata={"hnsw:space": "cosine", "embed_model": EMBED_MODEL},
    )

    ids = [c["chunk_id"] for c in chunks]
    texts = [c["text"] for c in chunks]
    metadatas = []
    for c in chunks:
        md = dict(c["metadata"])
        md["doc_id"] = c["doc_id"]
        md["chunk_id"] = c["chunk_id"]
        metadatas.append(md)

    bs = args.batch_size
    for i in tqdm(range(0, len(texts), bs), desc="embed+upsert"):
        batch_texts = texts[i : i + bs]
        emb = model.encode(
            batch_texts,
            batch_size=min(64, bs),
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        collection.add(
            ids=ids[i : i + bs],
            documents=batch_texts,
            metadatas=metadatas[i : i + bs],
            embeddings=emb.tolist(),
        )

    print(f"Done. Collection size: {collection.count()}")


if __name__ == "__main__":
    main()
