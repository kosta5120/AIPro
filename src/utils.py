"""Shared utilities: document loading, cleaning, chunking strategies."""
from __future__ import annotations

import json
import os
import re
from typing import Iterable, Iterator

URL_RE = re.compile(r"https?://\S+|www\.\S+")
WS_RE = re.compile(r"\s+")
SENT_SPLIT_RE = re.compile(r"(?<=[\.\!\?…])\s+|\n{2,}")


def clean_text(text: str, keep_urls: bool = False) -> tuple[str, list[str]]:
    """Normalize whitespace and optionally strip URLs.

    Returns (cleaned_text, urls_found).
    """
    if not text:
        return "", []
    urls = URL_RE.findall(text)
    if not keep_urls:
        text = URL_RE.sub(" ", text)
    text = text.replace("​", " ").replace("﻿", " ")
    text = WS_RE.sub(" ", text).strip()
    return text, urls


def load_pdf(path: str, max_pages: int | None = None) -> Iterator[dict]:
    """Load a PDF file and yield one doc per page, preserving page metadata.

    Pages are kept as separate docs so that page numbers survive into chunk
    metadata. The chunker downstream is responsible for splitting long pages;
    short pages (common in diagram-heavy PDFs) pass through as single chunks.
    """
    try:
        import fitz  # pymupdf
    except ImportError as e:
        raise ImportError(
            "pymupdf is required to load PDFs. Install it with: pip install pymupdf"
        ) from e

    filename = os.path.basename(path)
    base = os.path.splitext(filename)[0]
    base_clean = re.sub(r"[^a-zA-Z0-9_-]", "_", base)

    pdf = fitz.open(path)
    total_pages = len(pdf)
    page_limit = total_pages if max_pages is None else min(max_pages, total_pages)

    for page_num in range(page_limit):
        page = pdf[page_num]
        raw_text = page.get_text()
        text, _ = clean_text(raw_text)
        if not text:
            continue
        doc_id = f"pdf_{base_clean}_p{page_num + 1}"
        meta = {
            "source": filename,
            "page": page_num + 1,
            "total_pages": total_pages,
        }
        yield {"doc_id": doc_id, "text": text, "metadata": meta}


def load_pdf_full(path: str, max_pages: int | None = None) -> Iterator[dict]:
    """Load a PDF as one document per file (all pages concatenated).

    Use this when pages are individually too short to be split by the chunker
    (e.g. diagram-heavy textbooks where most pages are under 200 words).
    The chunker will then produce meaningfully different numbers of chunks
    for different chunk_size values, making ablation experiments valid.

    Page-number metadata is stored as 'page_start'/'page_end' covering the
    range of pages whose text ended up in each yielded document.
    """
    try:
        import fitz  # pymupdf
    except ImportError as e:
        raise ImportError(
            "pymupdf is required to load PDFs. Install it with: pip install pymupdf"
        ) from e

    filename = os.path.basename(path)
    base = os.path.splitext(filename)[0]
    base_clean = re.sub(r"[^a-zA-Z0-9_-]", "_", base)

    pdf = fitz.open(path)
    total_pages = len(pdf)
    page_limit = total_pages if max_pages is None else min(max_pages, total_pages)

    parts: list[str] = []
    first_page = 1
    last_page = 1

    for page_num in range(page_limit):
        page = pdf[page_num]
        raw_text = page.get_text()
        text, _ = clean_text(raw_text)
        if not text:
            continue
        if not parts:
            first_page = page_num + 1
        parts.append(text)
        last_page = page_num + 1

    if parts:
        full_text = " ".join(parts)
        doc_id = f"pdf_{base_clean}"
        meta = {
            "source": filename,
            "page_start": first_page,
            "page_end": last_page,
            "total_pages": total_pages,
        }
        yield {"doc_id": doc_id, "text": full_text, "metadata": meta}


def load_pdf_structured(path: str, max_pages: int | None = None) -> Iterator[dict]:
    """Extract a PDF into one document per logical section.

    Uses pymupdf's structured output and font-size heuristics to detect the
    document hierarchy (Module > Core Concept > Subsection). Each emitted
    document carries that section context in its metadata, so when chunks are
    later produced they inherit *which* section they came from — even if a
    long section gets sub-split into multiple word-level chunks.

    Heading detection (this PDF's font scheme):
      size >= 28        : Module marker ("Module 1")
      size 22-27        : Module title text ("Introduction to ML Paradigms")
      size 17-21        : Section heading ("Chapter Overview", "Core Concept: X",
                          "Module N Summary")
      size 13.5-16      : Named subsection ("Theoretical Foundation", etc.) —
                          only if the line text matches a known subsection
                          name; otherwise treated as body (math formula).
      size <= 11        : Body text
      size ~10 + digits : Page number (skipped)
    """
    try:
        import fitz  # pymupdf
    except ImportError as e:
        raise ImportError(
            "pymupdf is required to load PDFs. Install it with: pip install pymupdf"
        ) from e

    filename = os.path.basename(path)
    base = os.path.splitext(filename)[0]
    base_clean = re.sub(r"[^a-zA-Z0-9_-]", "_", base)

    pdf = fitz.open(path)
    total_pages = len(pdf)
    page_limit = total_pages if max_pages is None else min(max_pages, total_pages)

    KNOWN_SUBSECTIONS = {
        "Chapter Overview",
        "Theoretical Foundation",
        "Mathematical Intuition",
        "Python Implementation",
        "Engineering Best Practices",
        "Knowledge Check",
    }

    # Section state — emitted whenever a heading transition occurs.
    state = {
        "module": "",
        "concept": "",
        "subsection": "",
        "body": [],
        "page_start": 1,
        "page_end": 1,
    }
    section_index = 0

    def emit():
        nonlocal section_index
        text = " ".join(state["body"]).strip()
        text, _ = clean_text(text)
        state["body"] = []
        if not text:
            return None
        doc = {
            "doc_id": f"pdf_{base_clean}_sec{section_index}",
            "text": text,
            "metadata": {
                "source": filename,
                "section_index": section_index,
                "module": state["module"],
                "concept": state["concept"],
                "subsection": state["subsection"],
                "page_start": state["page_start"],
                "page_end": state["page_end"],
                "total_pages": total_pages,
            },
        }
        section_index += 1
        return doc

    def start_section(page_num: int) -> None:
        state["page_start"] = page_num
        state["page_end"] = page_num

    for page_num in range(page_limit):
        page = pdf[page_num]
        page_one_indexed = page_num + 1
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                spans = line.get("spans", [])
                if not spans:
                    continue
                line_text = " ".join(s["text"] for s in spans).strip()
                if not line_text:
                    continue
                max_size = max((s["size"] for s in spans), default=0)

                # Page number — skip.
                if max_size <= 10.5 and line_text.isdigit():
                    continue

                # Module marker (size 32).
                if max_size >= 28 and line_text.lower().startswith("module"):
                    doc = emit()
                    if doc:
                        yield doc
                    state["module"] = line_text
                    state["concept"] = ""
                    state["subsection"] = ""
                    start_section(page_one_indexed)
                    continue

                # Module title (size 24) — appends to module heading.
                if 22 <= max_size <= 27:
                    state["module"] = (state["module"] + " " + line_text).strip()
                    continue

                # Section heading (size 18). PDF headings sometimes wrap across
                # two visual lines (e.g., "Core Concept: Unsupervised" + "Learning").
                # If no body has accumulated since the last heading, treat this
                # line as a continuation of the previously-set concept/subsection
                # rather than a brand-new section.
                if max_size >= 17:
                    if not state["body"]:
                        if line_text.startswith("Core Concept"):
                            state["concept"] = line_text
                        elif state["concept"] and not line_text.startswith(
                            ("Chapter", "Module", "Knowledge")
                        ) and "Summary" not in line_text:
                            state["concept"] = (state["concept"] + " " + line_text).strip()
                        elif state["subsection"] and not line_text.startswith(
                            ("Chapter", "Core Concept", "Module", "Knowledge")
                        ):
                            state["subsection"] = (state["subsection"] + " " + line_text).strip()
                        else:
                            if line_text == "Chapter Overview":
                                state["subsection"] = "Chapter Overview"
                            elif "Summary" in line_text or line_text == "Knowledge Check":
                                state["subsection"] = line_text
                            else:
                                state["concept"] = line_text
                        continue
                    doc = emit()
                    if doc:
                        yield doc
                    if line_text.startswith("Core Concept"):
                        state["concept"] = line_text
                        state["subsection"] = ""
                    elif line_text == "Chapter Overview":
                        state["concept"] = ""
                        state["subsection"] = "Chapter Overview"
                    elif "Summary" in line_text or line_text == "Knowledge Check":
                        state["concept"] = ""
                        state["subsection"] = line_text
                    else:
                        state["concept"] = line_text
                        state["subsection"] = ""
                    start_section(page_one_indexed)
                    continue

                # Named subsection (size 14, bold) — only if it matches a known name.
                if max_size >= 13.5 and line_text in KNOWN_SUBSECTIONS:
                    if not state["body"]:
                        # Wrapped-heading continuation already handled above; here
                        # we simply update the subsection in place.
                        state["subsection"] = line_text
                        continue
                    doc = emit()
                    if doc:
                        yield doc
                    state["subsection"] = line_text
                    start_section(page_one_indexed)
                    continue

                # Everything else is body text (including size-14 math formulas).
                state["body"].append(line_text)
                state["page_end"] = page_one_indexed

    # Final flush.
    doc = emit()
    if doc:
        yield doc


def load_all_docs(
    raw_dir: str = "data/raw",
    max_per_source: int | None = None,
    mode: str = "full",
) -> list[dict]:
    """Load all PDF documents from raw_dir.

    mode="full"       : one document per PDF (all pages concatenated). Best when
                        ablating chunk size, since the chunker needs a long input
                        to produce meaningfully different chunk counts.
    mode="page"       : one document per page, preserving page-number metadata.
    mode="structured" : one document per logical section (Module > Core Concept >
                        Subsection), detected via font-size heuristics. Each
                        section carries section_index / module / concept /
                        subsection / page_start / page_end metadata that flows
                        through chunking, so each chunk knows where it came from.

    Returns a list of {doc_id, text, metadata} dicts.
    """
    loader_map = {
        "full":       load_pdf_full,
        "page":       load_pdf,
        "structured": load_pdf_structured,
    }
    if mode not in loader_map:
        raise ValueError(
            f"Unknown load mode {mode!r}; expected one of {sorted(loader_map)}"
        )
    loader = loader_map[mode]
    docs: list[dict] = []
    for fname in sorted(os.listdir(raw_dir)):
        if fname.lower().endswith(".pdf"):
            path = os.path.join(raw_dir, fname)
            docs.extend(loader(path, max_pages=max_per_source))
    return docs


def chunk_fixed(text: str, chunk_size: int = 500, overlap: int = 80) -> list[str]:
    """Fixed-size word-based chunking with overlap.

    chunk_size and overlap are measured in whitespace-separated tokens.
    """
    if not text:
        return []
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 4)
    step = chunk_size - overlap
    chunks = []
    for i in range(0, len(words), step):
        piece = words[i : i + chunk_size]
        if not piece:
            break
        chunks.append(" ".join(piece))
        if i + chunk_size >= len(words):
            break
    return chunks


def chunk_sentences(text: str, max_words: int = 120, overlap_sents: int = 1) -> list[str]:
    """Sentence-based chunking. Groups sentences until max_words is hit."""
    if not text:
        return []
    sents = [s.strip() for s in SENT_SPLIT_RE.split(text) if s and s.strip()]
    if not sents:
        return []
    chunks: list[str] = []
    cur: list[str] = []
    cur_words = 0
    i = 0
    while i < len(sents):
        s = sents[i]
        sw = len(s.split())
        if cur and cur_words + sw > max_words:
            chunks.append(" ".join(cur))
            keep = cur[-overlap_sents:] if overlap_sents > 0 else []
            cur = list(keep)
            cur_words = sum(len(x.split()) for x in cur)
            continue
        cur.append(s)
        cur_words += sw
        i += 1
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def chunk_recursive(text: str, chunk_size: int = 1500, overlap: int = 200) -> list[str]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    if not text:
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )
    return splitter.split_text(text)


def _breadcrumb(metadata: dict) -> str:
    parts = [metadata.get(k) for k in ("module", "concept", "subsection") if metadata.get(k)]
    return " > ".join(parts)


def chunk_document(
    doc: dict,
    strategy: str = "fixed",
    chunk_size: int = 1500,
    overlap: int = 200,
) -> list[dict]:
    """Chunk a single document and return chunk dicts with ids and metadata."""
    if strategy == "fixed":
        pieces = chunk_fixed(doc["text"], chunk_size=chunk_size, overlap=overlap)
    elif strategy == "sentence":
        pieces = chunk_sentences(doc["text"], max_words=max(40, chunk_size // 4), overlap_sents=1)
    elif strategy == "recursive":
        pieces = chunk_recursive(doc["text"], chunk_size=chunk_size, overlap=overlap)
    else:
        raise ValueError(f"Unknown chunk strategy: {strategy}; expected one of fixed, sentence, recursive")
    breadcrumb = _breadcrumb(doc.get("metadata", {}))
    out: list[dict] = []
    for i, piece in enumerate(pieces):
        if not piece.strip():
            continue
        text = f"[{breadcrumb}]\n\n{piece}" if breadcrumb else piece
        out.append(
            {
                "chunk_id": f"{doc['doc_id']}_chunk_{i}",
                "doc_id": doc["doc_id"],
                "text": text,
                "metadata": dict(doc["metadata"]),
            }
        )
    return out


def chunk_documents(
    docs: Iterable[dict],
    strategy: str = "fixed",
    chunk_size: int = 1500,
    overlap: int = 200,
) -> list[dict]:
    """Chunk every document and assign each emitted chunk a globally unique id."""
    chunks: list[dict] = []
    global_idx = 0
    for d in docs:
        for c in chunk_document(d, strategy, chunk_size, overlap):
            c["chunk_id"] = f"{c['doc_id']}_chunk_{global_idx}"
            chunks.append(c)
            global_idx += 1
    return chunks


def write_jsonl(path: str, rows: Iterable[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows
