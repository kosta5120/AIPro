"""Build a gold evaluation set from indexed chunks using a local LLM.

For every chunk in `data/processed/chunks.jsonl` the script asks the local
Ollama model (default `gemma3:4b`) to produce ONE evaluation example
grounded in that chunk's text. The resulting JSONL is written to
`eval/gold_set.jsonl` with one line per chunk.

Each line has exactly this shape:

    {
      "question": "...",
      "reference_answer": "...",
      "must_cite_chunk_ids": ["<chunk_id>"],
      "category": "factual" | "numerical" | "temporal" | "comparison" | "negation"
    }

Categories are cycled deterministically (factual, numerical, temporal,
comparison, negation) and the LLM is instructed to fall back to "factual"
when the chunk does not support the requested target category. For
"negation" items the reference_answer is forced to the canonical
"not found" sentence used by the RAG system.

Usage:
    python eval/build_gold_set.py                       # process all chunks
    python eval/build_gold_set.py --limit 20            # smoke test
    python eval/build_gold_set.py --resume              # continue an interrupted run
    python eval/build_gold_set.py --start 1000 --limit 500
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import shutil
import sys
from typing import Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from generation import call_ollama, OllamaUnavailable  # noqa: E402
from utils import read_jsonl  # noqa: E402

CATEGORIES = ["factual", "numerical", "temporal", "comparison", "negation"]
NOT_FOUND_ANSWER = "The information was not found in the provided context."

SYSTEM_PROMPT = (
    "You are a question-generation assistant for a PDF document RAG evaluation set.\n"
    "\n"
    "Given ONE passage from a PDF document (the context), produce EXACTLY ONE evaluation "
    "example as a JSON object with these fields:\n"
    "  - \"question\": a natural English question whose answer is entirely supported by "
    "the context (or, for negation, NOT supported by the context).\n"
    "  - \"reference_answer\": the short ground-truth answer derived from the context. "
    "For negation questions use exactly: \"The information was not found in the "
    "provided context.\"\n"
    "  - \"category\": one of \"factual\", \"numerical\", \"temporal\", \"comparison\", "
    "\"negation\". Prefer the requested target category if the context supports it; "
    "otherwise fall back to \"factual\".\n"
    "\n"
    "Rules:\n"
    "1. Build the question and answer ONLY from the provided context.\n"
    "2. Do not invent facts, numbers, or names not present in the context.\n"
    "3. Keep the question concise and self-contained (one sentence).\n"
    "4. The reference_answer must be short (1 sentence or a single value).\n"
    "5. For \"comparison\" questions, compare two concepts or values present in the "
    "context; otherwise fall back to \"factual\".\n"
    "6. For \"negation\" questions, ask whether the passage mentions a topic that is "
    "clearly NOT in its text; reference_answer must be the exact \"not found\" sentence.\n"
    "7. Return ONLY a single JSON object, no prose, no markdown fences."
)


def _format_chunk_context(chunk: dict) -> str:
    """Render a chunk as a small metadata header followed by its text body."""
    md = chunk.get("metadata", {}) or {}
    fields = [
        ("source", md.get("source")),
        ("page", md.get("page") or f"{md.get('page_start','?')}-{md.get('page_end','?')}"),
        ("module", md.get("module")),
        ("concept", md.get("concept")),
        ("subsection", md.get("subsection")),
        ("total_pages", md.get("total_pages")),
    ]
    header_lines = []
    for key, value in fields:
        if value in (None, ""):
            continue
        header_lines.append(f"{key}: {value}")
    header = "\n".join(header_lines)
    text = (chunk.get("text") or "").strip()
    body = f"text:\n{text}" if text else "text:"
    return f"{header}\n\n{body}".strip() if header else body


def _build_prompt(chunk: dict, target_category: str, strict_retry: bool = False) -> str:
    ctx = _format_chunk_context(chunk)
    extra = ""
    if strict_retry:
        extra = (
            "\n\nIMPORTANT: Your previous response could not be parsed as JSON. "
            "Output ONLY a single valid JSON object, with no prose and no markdown fences."
        )
    return (
        f"{SYSTEM_PROMPT}{extra}\n\n"
        f"Target category: {target_category}\n\n"
        f"Context:\n{ctx}\n\n"
        f"JSON:"
    )


_FENCE_OPEN_RE = re.compile(r"^```(?:json|JSON)?\s*", re.IGNORECASE)
_FENCE_CLOSE_RE = re.compile(r"\s*```\s*$")


def _strip_fences(s: str) -> str:
    s = (s or "").strip()
    s = _FENCE_OPEN_RE.sub("", s)
    s = _FENCE_CLOSE_RE.sub("", s)
    return s.strip()


def _extract_json_object(text: str) -> Optional[dict]:
    """Find the first balanced JSON object in `text` and return it as a dict.

    Tolerant of leading/trailing prose and ```json``` fences.
    Returns None on failure.
    """
    text = _strip_fences(text)
    start = text.find("{")
    if start == -1:
        return None
    decoder = json.JSONDecoder()
    try:
        obj, _ = decoder.raw_decode(text[start:])
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    return None


def _validate_item(raw: Optional[dict], chunk: dict) -> Optional[dict]:
    """Coerce the LLM dict into our canonical shape. Returns None if unusable."""
    if not isinstance(raw, dict):
        return None
    question = (raw.get("question") or "").strip()
    answer = (raw.get("reference_answer") or "").strip()
    if not question or not answer:
        return None
    category = raw.get("category")
    if category not in CATEGORIES:
        category = "factual"
    if category == "negation":
        answer = NOT_FOUND_ANSWER
        # Negation questions have no retrievable source chunk by definition —
        # leaving must_cite_chunk_ids empty tells the evaluator to skip them
        # in Hit@k / MRR@k and count them in the absent_questions bucket.
        cite_ids = []
    else:
        cite_ids = [chunk["chunk_id"]]
    return {
        "question": question,
        "reference_answer": answer,
        "must_cite_chunk_ids": cite_ids,
        "category": category,
    }


def _generate_one(
    chunk: dict,
    target_category: str,
    model: str,
    host: str,
    temperature: float,
    timeout: float,
) -> Optional[dict]:
    """Call the LLM up to twice; return a validated item or None on failure.

    Raises OllamaUnavailable if the LLM cannot be reached at all.
    """
    prompt = _build_prompt(chunk, target_category, strict_retry=False)
    text = call_ollama(
        prompt=prompt, model=model, host=host,
        temperature=temperature, timeout=timeout,
    )
    item = _validate_item(_extract_json_object(text), chunk)
    if item is not None:
        return item
    prompt2 = _build_prompt(chunk, target_category, strict_retry=True)
    text2 = call_ollama(
        prompt=prompt2, model=model, host=host,
        temperature=temperature, timeout=timeout,
    )
    return _validate_item(_extract_json_object(text2), chunk)


def _read_existing_ids_and_clean(path: str) -> set:
    """Return the set of chunk_ids already present in `path`.

    Also rewrites the file with only valid JSON lines and a final newline,
    so that subsequent append-mode writes do not corrupt the file if a
    previous run was interrupted mid-line.
    """
    ids: set = set()
    if not os.path.exists(path):
        return ids
    valid_lines: list = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            valid_lines.append(stripped)
            for cid in obj.get("must_cite_chunk_ids") or []:
                ids.add(cid)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        for line in valid_lines:
            f.write(line + "\n")
    os.replace(tmp_path, path)
    return ids


def _backup_if_needed(path: str) -> Optional[str]:
    """Copy `path` to `path + '.bak'` once, only if no backup exists yet."""
    if not os.path.exists(path):
        return None
    bak = path + ".bak"
    if os.path.exists(bak):
        return None
    shutil.copyfile(path, bak)
    return bak


def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a gold-set JSONL by prompting a local LLM (Ollama) "
            "with each chunk in the indexed corpus."
        ),
    )
    parser.add_argument(
        "--chunks-path",
        default=os.path.join(ROOT, "data", "processed", "chunks.jsonl"),
        help="Path to the indexed chunks JSONL (default: data/processed/chunks.jsonl).",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(ROOT, "eval", "gold_set.jsonl"),
        help="Output JSONL path (default: eval/gold_set.jsonl).",
    )
    parser.add_argument(
        "--model",
        default=os.environ.get("RAG_LLM_MODEL", "gemma3:4b"),
        help="Ollama model tag (default: gemma3:4b, env RAG_LLM_MODEL).",
    )
    parser.add_argument(
        "--ollama-host",
        default=os.environ.get("OLLAMA_HOST", "http://localhost:11434"),
        help="Ollama base URL (default: http://localhost:11434, env OLLAMA_HOST).",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Process only the first N chunks after --start (useful for smoke tests).",
    )
    parser.add_argument(
        "--start", type=int, default=0,
        help="Skip the first N chunks before processing.",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Append to --out and skip chunks already present in it.",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Seed for category rotation (deterministic but shuffleable).",
    )
    parser.add_argument(
        "--timeout", type=float, default=120.0,
        help="HTTP timeout per LLM call in seconds (default: 120).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)

    chunks = read_jsonl(args.chunks_path)
    if not chunks:
        print(f"No chunks found at {args.chunks_path}", file=sys.stderr)
        return 1
    print(f"Loaded {len(chunks)} chunks from {args.chunks_path}")

    sliced = chunks[args.start:]
    if args.limit is not None:
        sliced = sliced[: args.limit]
    print(
        f"Will process {len(sliced)} chunks "
        f"(start={args.start}, limit={args.limit})."
    )

    os.environ["OLLAMA_HOST"] = args.ollama_host

    bak = _backup_if_needed(args.out)
    if bak:
        print(f"Backed up existing {args.out} -> {bak}")

    already: set = set()
    if args.resume:
        already = _read_existing_ids_and_clean(args.out)
        print(f"Resume mode: {len(already)} chunk_ids already in {args.out}.")
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        out_f = open(args.out, "a", encoding="utf-8", buffering=1)
    else:
        out_dir = os.path.dirname(args.out)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        out_f = open(args.out, "w", encoding="utf-8", buffering=1)

    cats = list(CATEGORIES)
    random.Random(args.seed).shuffle(cats)
    print(f"Category rotation order (seed={args.seed}): {cats}")

    try:
        from tqdm import tqdm
        iterator = tqdm(list(enumerate(sliced)), total=len(sliced), desc="gold-set")
    except ImportError:
        iterator = enumerate(sliced)

    written = 0
    skipped_resume = 0
    skipped_parse = 0
    by_cat: dict = {}

    try:
        for offset, chunk in iterator:
            global_i = args.start + offset
            cid = chunk.get("chunk_id")
            if not cid:
                continue
            if cid in already:
                skipped_resume += 1
                continue

            target_cat = cats[global_i % len(cats)]
            try:
                item = _generate_one(
                    chunk=chunk,
                    target_category=target_cat,
                    model=args.model,
                    host=args.ollama_host,
                    temperature=args.temperature,
                    timeout=args.timeout,
                )
            except OllamaUnavailable as e:
                print(
                    f"\nOllama unavailable at {args.ollama_host}: {e}\n"
                    f"Stopping after writing {written} items. "
                    f"Existing output (if any) was backed up to {args.out + '.bak'}.",
                    file=sys.stderr,
                )
                return 2

            if item is None:
                skipped_parse += 1
                print(
                    f"\n[warn] Could not parse LLM output for {cid}; skipping.",
                    file=sys.stderr,
                )
                continue

            out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
            out_f.flush()
            written += 1
            by_cat[item["category"]] = by_cat.get(item["category"], 0) + 1
    finally:
        out_f.close()

    print()
    print(f"Wrote {written} items to {args.out}")
    if skipped_resume:
        print(f"Skipped (already in resume file): {skipped_resume}")
    if skipped_parse:
        print(f"Skipped (LLM parse failures):    {skipped_parse}")
    print("By category:")
    for cat in CATEGORIES:
        print(f"  {cat}: {by_cat.get(cat, 0)}")
    extras = {k: v for k, v in by_cat.items() if k not in CATEGORIES}
    if extras:
        print(f"  (unexpected categories: {extras})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
