"""Evaluate the RAG pipeline.

Computes Hit@k, MRR@k, and Answer-Accuracy (token F1 vs reference answers)
on `gold_set.jsonl`.  Runs an ablation across four configurations covering
two experiments:

  Experiment 1 — chunk size:  chunk_size=300 vs chunk_size=700 (k=5)
  Experiment 2 — top-k:       k=3 vs k=5 vs k=8 (chunk_size=300)

Answer accuracy is computed by generating answers with Ollama and measuring
token-level F1 against `reference_answer` in the gold set.  The metric is
silently skipped when Ollama is unavailable, and the table cell shows "N/A".

Usage:
    python eval/run_eval.py
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from utils import read_jsonl  # noqa: E402

GOLD_PATH    = os.path.join(ROOT, "eval", "gold_set.jsonl")
ANSWERS_PATH = os.path.join(ROOT, "eval", "answers_sample.jsonl")
RESULTS_DIR  = os.path.join(ROOT, "eval")

# ---------------------------------------------------------------------------
# Ablation configurations
# ---------------------------------------------------------------------------
# Each entry drives one row in the results table.
#
# Fields:
#   name        — collection / row label
#   retriever   — "dense" | "bm25" | "hybrid"
#   chunk_size  — words per chunk (kept constant in structured mode)
#   overlap     — words shared between consecutive chunks
#   k           — top-k chunks returned by retrieve()
#   reuse       — (optional) name of another config whose Chroma index +
#                 chunks file should be reused, avoiding a rebuild
#   notes       — free-form column shown in the results table
#
# Two experiments:
#   Experiment 1 — top-k variation (dense retrieval): k=3 vs k=5 vs k=8
#   Experiment 2 — retrieval strategy at k=5: dense vs BM25 vs hybrid (RRF)
ABLATION_CONFIGS = [
    # --- Experiment 1: top-k variation, dense retrieval ---
    # dense_k5 serves double duty: it's the middle point of Experiment 1 AND
    # the dense baseline of Experiment 2.
    {
        "name":       "dense_k3",
        "retriever":  "dense",
        "chunk_size": 1500,
        "overlap":    200,
        "k":          3,
        "notes":      "Dense, k=3 — focused but may miss evidence",
    },
    {
        "name":       "dense_k5",
        "retriever":  "dense",
        "chunk_size": 1500,
        "overlap":    200,
        "k":          5,
        "reuse":      "dense_k3",
        "notes":      "Dense, k=5 — baseline (Exp 1 mid-point, Exp 2 dense entry)",
    },
    {
        "name":       "dense_k8",
        "retriever":  "dense",
        "chunk_size": 1500,
        "overlap":    200,
        "k":          8,
        "reuse":      "dense_k3",
        "notes":      "Dense, k=8 — broader recall, more noise",
    },
    # --- Experiment 2: retrieval strategy at k=5 ---
    {
        "name":       "bm25_k5",
        "retriever":  "bm25",
        "chunk_size": 1500,
        "overlap":    200,
        "k":          5,
        "reuse":      "dense_k3",
        "notes":      "BM25 lexical — wins on named entities (BM25, RLHF, sklearn)",
    },
    {
        "name":       "hybrid_k5",
        "retriever":  "hybrid",
        "chunk_size": 1500,
        "overlap":    200,
        "k":          5,
        "reuse":      "dense_k3",
        "notes":      "Hybrid (dense + BM25, RRF fusion) — best of both",
    },
]

# Maximum gold-set items used for answer-accuracy sampling (keeps eval fast).
ANSWER_ACCURACY_SAMPLE = 20

# Load-mode used when (re)building ablation indices. Must match the load-mode
# the gold set was built against, otherwise chunk_ids won't match and every
# Hit@k will be 0.0.  Choices: "full" | "page" | "structured".
LOAD_MODE = "structured"


# ---------------------------------------------------------------------------
# Retrieval metrics
# ---------------------------------------------------------------------------

def doc_id_of(chunk_id: str) -> str:
    if "_chunk_" in chunk_id:
        return chunk_id.split("_chunk_")[0]
    return chunk_id


def evaluate_retrieval(retriever, gold: list[dict], k: int) -> dict:
    """Compute Hit@k and MRR@k.

    Items with empty ``must_cite_chunk_ids`` (negation / absence questions)
    are counted separately and excluded from Hit@k / MRR@k.
    """
    hits = 0
    mrr_sum = 0.0
    grounded = 0
    absent_total = 0

    for item in gold:
        gold_ids = set(item.get("must_cite_chunk_ids") or [])
        if not gold_ids:
            absent_total += 1
            continue
        grounded += 1
        results = retriever.retrieve(item["question"], k=k)
        rank = None
        for i, r in enumerate(results, 1):
            cid = r["chunk_id"]
            did = doc_id_of(cid)
            if cid in gold_ids or did in gold_ids:
                rank = i
                break
        if rank is not None:
            hits += 1
            mrr_sum += 1.0 / rank

    n = max(grounded, 1)
    return {
        "k":                   k,
        "grounded_questions":  grounded,
        "absent_questions":    absent_total,
        "hit_at_k":            hits / n,
        "mrr_at_k":            mrr_sum / n,
    }


# ---------------------------------------------------------------------------
# Answer-accuracy metric (token F1)
# ---------------------------------------------------------------------------

_STOP = frozenset(
    "a an the is are was were be been being have has had do does did "
    "will would could should may might shall can not no".split()
)


def _normalize(text: str) -> list[str]:
    """Lower-case, strip punctuation, remove stop-words."""
    tokens = re.sub(r"[^\w\s]", " ", text.lower()).split()
    return [t for t in tokens if t and t not in _STOP]


def _token_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1 between prediction and ground_truth strings."""
    pred  = set(_normalize(prediction))
    gold  = set(_normalize(ground_truth))
    if not pred or not gold:
        return 0.0
    common = pred & gold
    if not common:
        return 0.0
    precision = len(common) / len(pred)
    recall    = len(common) / len(gold)
    return 2 * precision * recall / (precision + recall)


def compute_answer_accuracy(
    retriever,
    gold: list[dict],
    k: int,
    n_sample: int = ANSWER_ACCURACY_SAMPLE,
) -> float | None:
    """Generate answers for up to *n_sample* grounded gold items and return
    the mean token F1 against their reference answers.

    Returns None if Ollama is unreachable.
    """
    from generation import generate_answer, OllamaUnavailable

    grounded = [g for g in gold if g.get("must_cite_chunk_ids")]
    sample   = grounded[:n_sample]
    scores: list[float] = []

    for item in sample:
        retrieved = retriever.retrieve(item["question"], k=k)
        try:
            gen = generate_answer(item["question"], retrieved)
            f1  = _token_f1(gen["answer_text"], item.get("reference_answer", ""))
            scores.append(f1)
        except OllamaUnavailable:
            return None   # abort early; caller will show N/A

    if not scores:
        return None
    return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# Index builder (subprocess for clean isolation)
# ---------------------------------------------------------------------------

def build_index(
    persist_dir: str,
    collection: str,
    chunk_size: int,
    overlap: int,
    max_per_source: int,
    load_mode: str = LOAD_MODE,
) -> None:
    if os.path.isdir(persist_dir):
        shutil.rmtree(persist_dir)
    cmd = [
        sys.executable,
        os.path.join(ROOT, "src", "build_index.py"),
        "--persist-dir",      persist_dir,
        "--collection-name",  collection,
        "--load-mode",        load_mode,
        "--chunk-strategy",   "recursive",
        "--chunk-size",       str(chunk_size),
        "--overlap",          str(overlap),
        "--max-per-source",   str(max_per_source),
        "--chunks-path",
        os.path.join(ROOT, "data", "processed", f"chunks_{collection}.jsonl"),
        "--clean",
    ]
    print("Running:", " ".join(cmd))
    subprocess.check_call(cmd)


def get_retriever_for(
    persist_dir: str,
    collection: str,
    retriever_type: str = "dense",
):
    """Return a retriever instance for the given ablation config.

    The chunks file path is derived from ``collection`` to match how
    ``build_index`` writes it (``data/processed/chunks_<collection>.jsonl``).
    """
    from retrieval import Retriever, BM25Retriever, HybridRetriever

    chunks_path = os.path.join(
        ROOT, "data", "processed", f"chunks_{collection}.jsonl"
    )
    if retriever_type == "dense":
        return Retriever(persist_dir=persist_dir, collection_name=collection)
    if retriever_type == "bm25":
        return BM25Retriever(chunks_path=chunks_path)
    if retriever_type == "hybrid":
        dense = Retriever(persist_dir=persist_dir, collection_name=collection)
        sparse = BM25Retriever(chunks_path=chunks_path)
        return HybridRetriever(dense=dense, sparse=sparse)
    raise ValueError(
        f"Unknown retriever type {retriever_type!r}; "
        f"expected one of: dense, bm25, hybrid"
    )


# ---------------------------------------------------------------------------
# Ablation runner
# ---------------------------------------------------------------------------

def run_ablation(max_per_source: int = 1500) -> list[dict]:
    """Build indices and evaluate all configs. Returns a list of result dicts."""
    gold         = read_jsonl(GOLD_PATH)
    results:list[dict] = []
    persist_root = os.path.join(ROOT, "data", "processed", "ablation")
    os.makedirs(persist_root, exist_ok=True)

    for cfg in ABLATION_CONFIGS:
        coll = cfg.get("reuse") or cfg["name"]
        pdir = os.path.join(persist_root, coll)
        retriever_type = cfg.get("retriever", "dense")

        # Build index only when not reusing a previously built one.
        if not cfg.get("reuse"):
            build_index(pdir, coll, cfg["chunk_size"], cfg["overlap"], max_per_source)

        retr = get_retriever_for(pdir, coll, retriever_type=retriever_type)

        # --- retrieval metrics ---
        metrics = evaluate_retrieval(retr, gold, k=cfg["k"])

        # --- answer accuracy ---
        print(f"  [answer-accuracy] sampling {ANSWER_ACCURACY_SAMPLE} items ...")
        ans_acc = compute_answer_accuracy(retr, gold, k=cfg["k"])

        row = {
            "config":              cfg["name"],
            "retriever":           retriever_type,
            "k":                   cfg["k"],
            "collection_size":     retr.count(),
            "grounded_questions":  metrics["grounded_questions"],
            "hit_at_k":            metrics["hit_at_k"],
            "mrr_at_k":            metrics["mrr_at_k"],
            "answer_accuracy":     ans_acc,
            "notes":               cfg.get("notes", ""),
        }
        results.append(row)
        print(json.dumps({k: v for k, v in row.items() if k != "notes"}, indent=2))

    return results


# ---------------------------------------------------------------------------
# Table formatter
# ---------------------------------------------------------------------------

def _fmt(val, decimals: int = 3) -> str:
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)


def format_table(rows: list[dict]) -> str:
    # In structured mode chunk_size / overlap are inactive (every section is
    # already shorter than any sensible chunk_size), so we omit them from the
    # report to avoid suggesting they were varied.
    headers = [
        "Experiment",
        "Retriever",
        "k",
        "collection_size",
        "Hit@k",
        "MRR@k",
        "Answer accuracy",
        "Notes",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for r in rows:
        cells = [
            r.get("config", ""),
            r.get("retriever", "dense"),
            _fmt(r.get("k")),
            _fmt(r.get("collection_size")),
            _fmt(r.get("hit_at_k")),
            _fmt(r.get("mrr_at_k")),
            _fmt(r.get("answer_accuracy")),
            r.get("notes", ""),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Answer sampling (saved to answers_sample.jsonl)
# ---------------------------------------------------------------------------

def sample_answers(retriever, gold: list[dict], n: int = 10) -> None:
    """Generate answers for n questions and save to ANSWERS_PATH.
    Gracefully skips if Ollama is unreachable.
    """
    from generation import generate_answer, OllamaUnavailable

    rows    = []
    skipped = False
    for item in gold[:n]:
        retrieved = retriever.retrieve(item["question"], k=5)
        try:
            gen      = generate_answer(item["question"], retrieved)
            ans_text = gen["answer_text"]
            cited    = gen["cited_ids"]
        except OllamaUnavailable as e:
            print(f"[answers] Ollama unavailable, skipping generation: {e}")
            skipped = True
            break
        rows.append(
            {
                "question":            item["question"],
                "category":            item.get("category"),
                "reference_answer":    item.get("reference_answer"),
                "must_cite_chunk_ids": item.get("must_cite_chunk_ids"),
                "answer":              ans_text,
                "cited":               cited,
                "retrieved_ids":       [r["chunk_id"] for r in retrieved],
            }
        )
    if rows:
        with open(ANSWERS_PATH, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Wrote {len(rows)} generated answers to {ANSWERS_PATH}")
    elif skipped:
        print("No generated answers produced (Ollama unavailable).")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run the RAG ablation evaluation.")
    parser.add_argument(
        "--clean",
        action="store_true",
        help=(
            "Wipe data/processed/ablation/ before running. Use this after "
            "changing the embed model, load mode, or chunk strategy."
        ),
    )
    args = parser.parse_args()

    if not os.path.exists(GOLD_PATH):
        print(f"Gold set missing at {GOLD_PATH}. Run: python eval/build_gold_set.py")
        sys.exit(1)

    if args.clean:
        ablation_root = os.path.join(ROOT, "data", "processed", "ablation")
        if os.path.isdir(ablation_root):
            print(f"Cleaning {ablation_root}")
            shutil.rmtree(ablation_root)

    rows  = run_ablation()
    table = format_table(rows)

    print("\n=== Ablation results ===")
    print(table)

    timestamp   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(RESULTS_DIR, f"results_{timestamp}.md")

    with open(results_path, "w", encoding="utf-8") as f:
        f.write(f"# RAG Ablation Study — {timestamp}\n\n")
        f.write(
            f"**Load mode:** `{LOAD_MODE}` "
            "(one document per Module > Concept > Subsection section; "
            "chunks inherit section metadata and prepend a breadcrumb to their text)\n\n"
        )

        f.write("## Experiment 1 — Top-k: 3 vs 5 vs 8 (dense retrieval)\n\n")
        f.write(
            "Keeps the dense index identical and varies only how many chunks "
            "are passed to the LLM, trading recall against context noise.\n\n"
        )

        f.write("## Experiment 2 — Retrieval strategy: dense vs BM25 vs hybrid (k=5)\n\n")
        f.write(
            "Same chunk corpus, three different retrieval algorithms:\n"
            "- **dense**: cosine similarity over `bge-large-en-v1.5` embeddings (semantic match)\n"
            "- **BM25**: lexical match with term-frequency saturation (wins on named entities)\n"
            "- **hybrid**: Reciprocal Rank Fusion of dense + BM25 top-20 pools (no score "
            "normalization needed)\n\n"
        )

        f.write(
            "*Note on chunking:* chunks are produced by LangChain's "
            "`RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=200)` "
            "operating on each section produced by the structured PDF loader, "
            "with a [Module > Concept > Subsection] breadcrumb prepended to each chunk's text.\n\n"
        )

        f.write("## Results\n\n")
        f.write(table + "\n\n")
        f.write(
            "> **Answer accuracy** = mean token-level F1 between the generated answer "
            f"and the reference answer, sampled over the first "
            f"{ANSWER_ACCURACY_SAMPLE} grounded gold-set items.\n"
            "> `N/A` means Ollama was unavailable during evaluation.\n"
        )
    print(f"\nWrote {results_path}")

    # Save a sample of full answers from the baseline config.
    try:
        from retrieval import Retriever

        baseline = ABLATION_CONFIGS[0]
        baseline_collection = baseline.get("reuse") or baseline["name"]
        retr = Retriever(
            persist_dir=os.path.join(
                ROOT, "data", "processed", "ablation", baseline_collection
            ),
            collection_name=baseline_collection,
        )
        gold = read_jsonl(GOLD_PATH)
        sample_answers(retr, gold, n=10)
    except Exception as e:
        print(f"[answers] Skipped: {e}")


if __name__ == "__main__":
    main()
