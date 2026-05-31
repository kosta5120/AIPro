"""RAG system entry point. Exposes `answer(question: str) -> dict`."""
from __future__ import annotations

import os
import re
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from retrieval import RerankingRetriever, RERANK_TOP_N  # noqa: E402
from generation import generate_answer, OllamaUnavailable  # noqa: E402

DEFAULT_TOP_K = int(os.environ.get("RAG_TOP_K", str(RERANK_TOP_N)))

_retriever: Optional[RerankingRetriever] = None


def get_retriever() -> RerankingRetriever:
    global _retriever
    if _retriever is None:
        _retriever = RerankingRetriever()
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
            "answer": "I do not know",
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


# ---------------------------------------------------------------------------
# CLI report formatting (stdlib only, UTF-8 Markdown output)
# ---------------------------------------------------------------------------

_PIPELINE_DESC = "Hybrid (dense bge + BM25, k=20) -> cross-encoder rerank -> LLM"
_CITED_RE = re.compile(r"\[([^\[\]]+_chunk_\d+)\]")
_SHORT_CID_RE = re.compile(r"(sec\d+_chunk_\d+)$")
_BREADCRUMB_RE = re.compile(r"^\s*\[[^\]]*\]\s*")
_MODULE_PREFIX_RE = re.compile(r"^(Module\s+\d+)\b")
_CORE_CONCEPT_PREFIX = "core concept:"


def _truncate(s: str, n: int) -> str:
    s = s or ""
    if n <= 3:
        return s[:n]
    return s if len(s) <= n else s[: n - 3] + "..."


def _short_chunk_id(chunk_id: str) -> str:
    """Return the trailing ``sec<N>_chunk_<M>`` portion of a chunk id.

    Falls back to the last three underscore-separated segments when the
    standard pattern is absent, so unrelated id shapes still render
    something reasonable.
    """
    cid = chunk_id or ""
    m = _SHORT_CID_RE.search(cid)
    if m:
        return m.group(1)
    parts = cid.split("_")
    return "_".join(parts[-3:]) if len(parts) >= 3 else cid


def _escape_cell(s: str) -> str:
    """Make ``s`` safe to drop into a Markdown table cell.

    Escapes literal ``|`` characters and collapses any whitespace
    (including newlines) into single spaces so the cell stays on one line.
    """
    s = (s or "").replace("|", "\\|")
    return " ".join(s.split())


def _strip_core_concept(concept: str) -> str:
    concept = (concept or "").strip()
    if concept.lower().startswith(_CORE_CONCEPT_PREFIX):
        concept = concept[len(_CORE_CONCEPT_PREFIX):].strip()
    return concept


def _module_section_label(meta: dict, max_len: int = 50) -> str:
    """Compact ``Module N · Concept · Subsection`` label for the table.

    - Compresses ``Module N <long title>`` to just ``Module N`` so the
      cell stays scannable.
    - Strips a leading ``Core Concept:`` prefix from ``concept``.
    - Joins the non-empty parts with `` · `` and truncates to ``max_len``.
    """
    meta = meta or {}
    module = (meta.get("module") or "").strip()
    concept = _strip_core_concept(meta.get("concept") or "")
    subsection = (meta.get("subsection") or "").strip()

    m = _MODULE_PREFIX_RE.match(module)
    if m:
        module = m.group(1)

    parts = [p for p in (module, concept, subsection) if p]
    label = " \u00b7 ".join(parts) if parts else "-"
    return _truncate(label, max_len)


def _pages(meta: dict) -> str:
    """Format pages as ``61`` or ``61-62``; ``-`` when missing."""
    ps, pe = meta.get("page_start"), meta.get("page_end")
    if ps is None and pe is None:
        return "-"
    if ps is None:
        return f"{pe}"
    if pe is None or pe == ps:
        return f"{ps}"
    return f"{ps}-{pe}"


def _snippet(text: str, max_len: int = 90) -> str:
    """Short single-line preview of chunk text.

    Strips the leading ``[Module ... > ... > ...]`` breadcrumb header that
    every chunk carries (the breadcrumb is already surfaced in the
    Module/Section column), collapses whitespace, and truncates with an
    ellipsis.
    """
    s = text or ""
    s = _BREADCRUMB_RE.sub("", s, count=1)
    s = " ".join(s.split())
    if len(s) <= max_len:
        return s
    return s[: max_len - 3].rstrip() + "..."


def _short_label(meta: dict) -> str:
    """One-line label for the cited-sources list.

    Prefers ``subsection``; falls back to ``concept`` (with the
    ``Core Concept:`` prefix removed), then ``-``.
    """
    meta = meta or {}
    subsection = (meta.get("subsection") or "").strip()
    if subsection:
        return subsection
    concept = _strip_core_concept(meta.get("concept") or "")
    if concept:
        return concept
    return "-"


def _extract_cited(answer_text: str) -> list[str]:
    """Return [chunk_id] tokens in answer order, deduplicated."""
    seen: list[str] = []
    seen_set: set[str] = set()
    for m in _CITED_RE.finditer(answer_text or ""):
        cid = m.group(1)
        if cid not in seen_set:
            seen_set.add(cid)
            seen.append(cid)
    return seen


def format_report(query: str, result: dict) -> str:
    """Render a Markdown report for an ``answer()`` result.

    Emits a real Markdown document (H1, H2 headings, pipe tables with
    separator rows, backticked chunk ids, numbered cited-sources list).
    Renders correctly in any Markdown preview (Cursor / VS Code / GitHub);
    the Answer body is inserted verbatim so ``[chunk_id]`` citation tokens
    stay intact.
    """
    answer_text = (result.get("answer") or "").strip()
    retrieved = result.get("retrieved_chunks") or []
    cited_ids = _extract_cited(answer_text)
    by_id = {c.get("chunk_id"): c for c in retrieved}

    out: list[str] = []

    out.append("# RAG Query Report")
    out.append("")

    out.append("## Run summary")
    out.append("")
    out.append("| Field | Value |")
    out.append("|-------|-------|")
    out.append(f"| Query | {_escape_cell(query)} |")
    out.append(f"| Pipeline | {_escape_cell(_PIPELINE_DESC)} |")
    out.append(f"| Retrieved chunks | {len(retrieved)} |")
    out.append(f"| Cited in answer | {len(cited_ids)} |")
    out.append("")

    out.append("## Answer")
    out.append("")
    out.append(answer_text if answer_text else "(no answer)")
    out.append("")

    out.append("## Cited sources")
    out.append("")
    if not cited_ids:
        out.append("(none -- model did not cite any [chunk_id] tokens)")
    else:
        for i, cid in enumerate(cited_ids, 1):
            meta = (by_id.get(cid) or {}).get("metadata") or {}
            short = _short_chunk_id(cid)
            out.append(f"{i}. `{short}` \u2014 {_short_label(meta)}")
    out.append("")

    out.append("## Top retrieved chunks (after reranking)")
    out.append("")
    if not retrieved:
        out.append("(no chunks retrieved)")
    else:
        out.append("| # | Chunk ID | Module / Section | Pages | What it contains |")
        out.append("|---|----------|------------------|-------|-------------------|")
        for i, c in enumerate(retrieved, 1):
            meta = c.get("metadata") or {}
            short_id = _short_chunk_id(c.get("chunk_id", ""))
            section = _escape_cell(_module_section_label(meta))
            pages = _escape_cell(_pages(meta))
            snip = _escape_cell(_snippet(c.get("text", "")))
            out.append(
                f"| {i} | `{short_id}` | {section} | {pages} | {snip} |"
            )
    out.append("")

    return "\n".join(out)


if __name__ == "__main__":
    import argparse
    import json
    from datetime import datetime

    parser = argparse.ArgumentParser(
        description=(
            "Ask the RAG pipeline a question against the indexed corpus. "
            "By default the formatted Markdown report is written to "
            "eval/rag_runs/rag_<YYYYMMDD_HHMMSS>.md and only the saved "
            "file path is printed to stdout."
        ),
    )
    parser.add_argument(
        "query",
        nargs="*",
        help="Query text. Multiple tokens are joined with spaces.",
    )
    parser.add_argument(
        "--json",
        dest="as_json",
        action="store_true",
        help=(
            "Print the raw answer dict as JSON to stdout. "
            "Skips writing the Markdown file."
        ),
    )
    parser.add_argument(
        "--out",
        dest="out_path",
        default=None,
        help=(
            "Override the Markdown output file path. "
            "Default: eval/rag_runs/rag_<YYYYMMDD_HHMMSS>.md."
        ),
    )
    parser.add_argument(
        "--print",
        dest="also_print",
        action="store_true",
        help="Also print the Markdown report to stdout after saving the file.",
    )
    args = parser.parse_args()

    q = " ".join(args.query) or "What is the main topic of this document?"
    result = answer(q)

    if args.as_json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        report = format_report(q, result)

        if args.out_path:
            out_path = args.out_path
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = os.path.join("eval", "rag_runs", f"rag_{ts}.md")

        out_abs = os.path.abspath(out_path)
        os.makedirs(os.path.dirname(out_abs), exist_ok=True)

        with open(out_abs, "w", encoding="utf-8") as f:
            f.write(report)
            if not report.endswith("\n"):
                f.write("\n")

        print(f"Saved report to: {out_abs}")

        if args.also_print:
            print(report)
