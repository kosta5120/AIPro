"""LLM answer generation using Ollama (gemma3:4b)."""
from __future__ import annotations

import os
import re
from typing import Optional

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = os.environ.get("RAG_LLM_MODEL", "gemma3:4b")

SYSTEM_PROMPT = (
    "You are a careful question-answering assistant for a PDF document corpus. "
    "Follow these rules strictly:\n"
    "1. Answer ONLY using the provided context chunks.\n"
    "2. If the answer is not present in the context, reply exactly: "
    "'The information was not found in the provided context.'\n"
    "3. Cite the chunks you used inline using their chunk_id in square brackets, "
    "e.g. [pdf_my_document_p5_chunk_0].\n"
    "4. Do not invent facts, numbers, or names not present in the context.\n"
    "5. Keep the answer concise (1-4 sentences) unless the question requires a list."
)

CITATION_RE = re.compile(r"\[([^\[\]\s,]+_chunk_\d+)\]")


def _format_context(chunks: list[dict]) -> str:
    lines = []
    for c in chunks:
        md = c.get("metadata", {}) or {}
        page_info = (
            f"p{md['page']}" if "page" in md
            else f"p{md.get('page_start', '?')}-{md.get('page_end', '?')}"
        )
        # Build a breadcrumb when structured-mode metadata is present.
        crumbs = [
            md.get(k) for k in ("module", "concept", "subsection") if md.get(k)
        ]
        section_path = " > ".join(crumbs) if crumbs else ""
        header_parts = [f"[{c['chunk_id']}]"]
        if section_path:
            header_parts.append(f"section={section_path!r}")
        header_parts.append(f"pages={page_info}")
        header = "(" + ", ".join(header_parts[1:]) + ")"
        lines.append(f"{header_parts[0]} {header}\n{c['text']}")
    return "\n\n".join(lines)


def build_prompt(question: str, chunks: list[dict]) -> str:
    ctx = _format_context(chunks)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Context:\n{ctx}\n\n"
        f"Question: {question}\n\n"
        f"Answer (with [chunk_id] citations):"
    )


def parse_citations(answer_text: str, retrieved_ids: list[str]) -> list[str]:
    found = CITATION_RE.findall(answer_text or "")
    retrieved_set = set(retrieved_ids)
    seen = set()
    cited = []
    for cid in found:
        if cid in retrieved_set and cid not in seen:
            cited.append(cid)
            seen.add(cid)
    return cited


class OllamaUnavailable(RuntimeError):
    pass


def call_ollama(
    prompt: str,
    model: str = DEFAULT_MODEL,
    host: str = OLLAMA_HOST,
    temperature: float = 0.2,
    timeout: float = 120.0,
) -> str:
    """Send `prompt` to a local Ollama server and return the raw response text.

    Tries the `ollama` Python package first, then falls back to a plain HTTP
    POST against `/api/generate`. Raises `OllamaUnavailable` if neither works.
    """
    try:
        import ollama  # type: ignore

        client = ollama.Client(host=host)
        resp = client.generate(
            model=model, prompt=prompt, options={"temperature": temperature}
        )
        return resp.get("response", "") or ""
    except Exception as e_pkg:
        try:
            import requests

            r = requests.post(
                f"{host}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": temperature},
                },
                timeout=timeout,
            )
            r.raise_for_status()
            return r.json().get("response", "") or ""
        except Exception as e_http:
            raise OllamaUnavailable(
                f"Ollama not reachable at {host} "
                f"(pkg error: {e_pkg!r}; http error: {e_http!r})"
            )


def _call_ollama(prompt: str, model: str = DEFAULT_MODEL, timeout: float = 120.0) -> str:
    return call_ollama(
        prompt=prompt,
        model=model,
        host=OLLAMA_HOST,
        temperature=0.2,
        timeout=timeout,
    )


def generate_answer(
    question: str, chunks: list[dict], model: Optional[str] = None
) -> dict:
    """Run the LLM and return {answer_text, cited_ids}.

    Raises OllamaUnavailable if Ollama is not reachable.
    """
    prompt = build_prompt(question, chunks)
    text = _call_ollama(prompt, model=model or DEFAULT_MODEL)
    text = (text or "").strip()
    cited = parse_citations(text, [c["chunk_id"] for c in chunks])
    return {"answer_text": text, "cited_ids": cited}
