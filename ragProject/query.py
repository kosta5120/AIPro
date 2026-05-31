"""Ask a question. Embeds the query, retrieves top-K chunks from the
MongoDB `rag_chunks` collection, and asks an LLM to answer using only
those chunks.

If Ollama is unavailable, falls back to printing the retrieved chunks only.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import statistics
import sys
import textwrap

# Force UTF-8 stdout so Hebrew/Arabic post text prints on Windows consoles.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

import numpy as np
from pymongo import MongoClient
from pymongo.collection import Collection
from sentence_transformers import SentenceTransformer

from config import (
    AUTO_INGEST,
    EMBEDDING_MODEL,
    MONGO_CHUNKS_COLLECTION,
    MONGO_COLLECTIONS,
    MONGO_DB,
    MONGO_URI,
    OLLAMA_HOST,
    OLLAMA_MODEL,
    TEMPERATURE,
    TOP_K,
)
from ingest import SOURCE_MAP, run_ingest


SYSTEM_PROMPT = (
    "You answer questions using the provided social-media posts.\n"
    "Rules:\n"
    "1. For every relevant post, quote or paraphrase the key content in 1-2 sentences.\n"
    "2. After each piece of information, cite the post number in square brackets like [1] or [2,3].\n"
    "3. Do NOT answer with citations alone — always include the actual content.\n"
    "4. If the posts don't answer the question, say so explicitly."
)


# Multilingual few-shot examples covering all three classes.
# These are shown to the LLM as labelled samples before it classifies the retrieved posts.
FEW_SHOT_SENTIMENT_EXAMPLES = [
    ("I love the new update, finally a great product!", "positive"),
    ("Best day of my life, can't stop smiling 😊", "positive"),
    ("יום מקסים בים, השמש נהדרת", "positive"),                      # Hebrew
    ("Terrible service, totally disappointed and angry.", "negative"),
    ("This is awful, worst experience ever.", "negative"),
    ("الطقس سيء جداً اليوم ولا أستطيع الخروج", "negative"),         # Arabic
    ("The package arrived today.", "neutral"),
    ("Meeting moved to 3pm tomorrow.", "neutral"),
    ("@user check this link https://example.com", "neutral"),
]


RAG_META_COLLECTION = "_rag_meta"


def maybe_auto_ingest(model: SentenceTransformer) -> None:
    """Cheap "is there new data?" check; runs run_ingest() only if needed.

    For each source collection in MONGO_COLLECTIONS we compare:
      - source_coll.estimated_document_count()   (O(1) — uses collStats)
      - a high-water mark cached in `_rag_meta` (tiny, one doc per platform),
        falling back to len(chunks.distinct("source_id", {"source": platform}))
        when the cache is stale or missing.

    If every platform already has at least as many distinct source_ids in
    rag_chunks (or its cached high-water mark) as documents in its source
    collection, we skip ingestion entirely — no embedding, no index load yet.

    After any ingest run (even no-op ones, e.g. all "new" posts had empty or
    byte-duplicate text), we refresh `_rag_meta` so the next query doesn't
    keep re-triggering ingest forever.
    """
    if not AUTO_INGEST:
        return

    print("Checking MongoDB for new posts...")
    client = MongoClient(MONGO_URI)
    try:
        db = client[MONGO_DB]
        chunks = db[MONGO_CHUNKS_COLLECTION]
        meta = db[RAG_META_COLLECTION]

        name_width = max((len(n) for n in MONGO_COLLECTIONS), default=0) + 1
        needs_ingest = False
        checked: list[tuple[str, int]] = []  # (platform, src_n) to refresh after ingest

        for source_name in MONGO_COLLECTIONS:
            mapping = SOURCE_MAP.get(source_name)
            if mapping is None:
                print(f"  {source_name:<{name_width}} (no mapping, skipped)")
                continue
            platform = mapping["platform"]
            source_coll = db[source_name]
            src_n = source_coll.estimated_document_count()
            checked.append((platform, src_n))

            cached_doc = meta.find_one(
                {"platform": platform}, {"last_seen_source_count": 1}
            )
            cached = cached_doc.get("last_seen_source_count") if cached_doc else None

            if cached is not None and cached >= src_n:
                print(
                    f"  {source_name + ':':<{name_width + 1}} "
                    f"{src_n} in source, {cached} already indexed  -> up to date"
                )
                continue

            indexed = len(chunks.distinct("source_id", {"source": platform}))
            if indexed >= src_n:
                print(
                    f"  {source_name + ':':<{name_width + 1}} "
                    f"{src_n} in source, {indexed} already indexed  -> up to date"
                )
                meta.update_one(
                    {"platform": platform},
                    {"$set": {"last_seen_source_count": src_n}},
                    upsert=True,
                )
                continue

            print(
                f"  {source_name + ':':<{name_width + 1}} "
                f"{src_n} in source, {indexed} already indexed  "
                f"-> {src_n - indexed} new posts to ingest"
            )
            needs_ingest = True

        if not needs_ingest:
            total = chunks.estimated_document_count()
            print(f"All sources up to date ({total} chunks indexed).")
            return
    finally:
        client.close()

    print("...ingesting...")
    run_ingest(model=model)

    client = MongoClient(MONGO_URI)
    try:
        meta = client[MONGO_DB][RAG_META_COLLECTION]
        for platform, src_n in checked:
            meta.update_one(
                {"platform": platform},
                {"$set": {"last_seen_source_count": src_n}},
                upsert=True,
            )
    finally:
        client.close()


def load_index(chunks: Collection) -> tuple[list[dict], np.ndarray]:
    meta: list[dict] = []
    vecs: list[np.ndarray] = []
    projection = {
        "source": 1,
        "source_collection": 1,
        "source_id": 1,
        "natural_id": 1,
        "author": 1,
        "created_date": 1,
        "content": 1,
        "embedding": 1,
    }
    for d in chunks.find({}, projection):
        emb = d.get("embedding")
        if not emb:
            continue
        vecs.append(np.frombuffer(bytes(emb), dtype=np.float32))
        meta.append(
            {
                "id": d.get("_id"),
                "source": d.get("source"),
                "source_collection": d.get("source_collection"),
                "source_id": d.get("source_id"),
                "natural_id": d.get("natural_id"),
                "author": d.get("author"),
                "date": d.get("created_date"),
                "content": d.get("content") or "",
            }
        )
    if not meta:
        return [], np.zeros((0, 0), dtype=np.float32)
    return meta, np.vstack(vecs)


def retrieve(query: str, model: SentenceTransformer, meta, mat, k: int = TOP_K):
    q = model.encode([query], normalize_embeddings=True)
    q = np.asarray(q, dtype=np.float32)[0]
    # rows are normalized -> cosine == dot product
    scores = mat @ q
    top_idx = np.argsort(-scores)[:k]
    return [(float(scores[i]), meta[i]) for i in top_idx]


CITATION_RE = re.compile(r"(\[\d+(?:\s*,\s*\d+)*\])")


def _supports_color() -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if not sys.stdout.isatty():
        return False
    if sys.platform == "win32":
        return os.environ.get("WT_SESSION") or os.environ.get("ANSICON") or "TERM" in os.environ
    return True


_USE_COLOR = _supports_color()


def _c(text: str, code: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _USE_COLOR else text


def _term_width(default: int = 100) -> int:
    try:
        w = shutil.get_terminal_size((default, 20)).columns
        return max(60, min(w, 120))
    except Exception:
        return default


def print_section_header(title: str) -> None:
    width = _term_width()
    bar = "─" * width
    print()
    print(_c(bar, "2;36"))
    print(_c(f" {title}", "1;36"))
    print(_c(bar, "2;36"))


def pretty_print_answer(answer: str) -> None:
    """Render an LLM answer that contains [N] / [N,M] citations as readable blocks.

    Each citation marker starts its own indented paragraph so the analyst can
    scan post-by-post instead of reading a wall of wrapped text.
    """
    width = _term_width()
    answer = (answer or "").strip()
    if not answer:
        print(_c("(empty answer)", "2"))
        return

    parts = CITATION_RE.split(answer)
    # parts looks like: [leading_text, '[1]', text1, '[2]', text2, ...]

    if len(parts) == 1:
        print(textwrap.fill(answer, width))
        return

    leading = parts[0].strip()
    if leading:
        print(textwrap.fill(leading, width))
        print()

    for i in range(1, len(parts), 2):
        cite = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        body = re.sub(r"\s+", " ", body)
        print(_c(cite, "1;33"))
        if body:
            print(
                textwrap.fill(
                    body,
                    width,
                    initial_indent="    ",
                    subsequent_indent="    ",
                )
            )
        print()


def pretty_print_sources(hits, sentiments) -> None:
    """Print retrieved posts one per block, wrapping the content for readability."""
    width = _term_width()
    body_width = max(40, width - 4)
    sentiment_color = {"positive": "1;32", "negative": "1;31", "neutral": "1;37"}

    for rank, ((score, m), s) in enumerate(zip(hits, sentiments), 1):
        date = m["date"].strftime("%Y-%m-%d") if m["date"] else "?"
        sent = s.get("sentiment", "?")
        conf = s.get("confidence", 0.0)
        head_left = _c(f"[{rank}]", "1;33")
        head_meta = _c(
            f"{m['source']} #{m['source_id']} | {m['author'] or '?'} | {date} | score={score:.3f}",
            "2",
        )
        head_sent = _c(
            f"{sent} ({conf:.2f})",
            sentiment_color.get(sent, "2"),
        )
        print(f"{head_left} {head_meta}  sentiment={head_sent}")
        content = (m["content"] or "").strip()
        if content:
            for line in content.splitlines() or [""]:
                line = line.strip()
                if not line:
                    print()
                    continue
                print(
                    textwrap.fill(
                        line,
                        body_width,
                        initial_indent="    ",
                        subsequent_indent="    ",
                    )
                )
        print()


def format_context(hits) -> str:
    parts = []
    for rank, (score, m) in enumerate(hits, 1):
        date = m["date"].strftime("%Y-%m-%d") if m["date"] else "?"
        head = f"[{rank}] {m['source']} #{m['source_id']} | {m['author'] or '?'} | {date} | score={score:.3f}"
        parts.append(head + "\n" + (m["content"] or "").strip())
    return "\n\n".join(parts)


def _build_sentiment_prompt(hits) -> str:
    examples = "\n\n".join(
        f"POST: {text}\nSENTIMENT: {label}"
        for text, label in FEW_SHOT_SENTIMENT_EXAMPLES
    )
    posts = "\n\n".join(
        f"POST {i+1}: {(m['content'] or '').strip()[:400]}"
        for i, (_, m) in enumerate(hits)
    )
    return (
        "You are a sentiment classifier. Classify EACH numbered POST as "
        '"positive", "negative", or "neutral".\n'
        "Reply with ONLY a JSON object in this exact shape — one entry per post, "
        "covering ALL posts:\n"
        '{"results":['
        '{"post":1,"sentiment":"positive","confidence":0.92},'
        '{"post":2,"sentiment":"neutral","confidence":0.78}'
        "]}\n"
        "Confidence must be a float between 0.0 and 1.0 reflecting how certain you are.\n\n"
        f"FEW-SHOT EXAMPLES:\n{examples}\n\n"
        f"POSTS TO CLASSIFY:\n{posts}\n\n"
        "JSON:"
    )


def ollama_available() -> bool:
    """Return True if the Ollama server responds."""
    try:
        import ollama

        ollama.Client(host=OLLAMA_HOST).list()
        return True
    except Exception:
        return False


def _llm_complete(prompt: str) -> str:
    """Send a prompt to Ollama (gemma3:4b by default), return text."""
    import ollama

    client = ollama.Client(host=OLLAMA_HOST)
    resp = client.chat(
        model=OLLAMA_MODEL,
        messages=[{"role": "user", "content": prompt}],
        format="json",
    )
    return resp["message"]["content"]


def classify_sentiments(hits) -> list[dict]:
    """Few-shot sentiment classification of all hits in one LLM call.
    Returns a list aligned with hits: [{'sentiment': str, 'confidence': float}, ...]"""
    if not hits:
        return []
    fallback = [{"sentiment": "?", "confidence": 0.0} for _ in hits]
    try:
        raw = _llm_complete(_build_sentiment_prompt(hits))
    except Exception as e:
        print(f"  (sentiment classification failed: {e})")
        return fallback
    if not raw:
        return fallback

    import json
    import re

    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if not match:
            return fallback
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            return fallback

    # Some providers wrap the array in {"results": [...]} or {"posts": [...]}.
    if isinstance(data, dict):
        for key in ("results", "posts", "classifications", "data"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break

    result = list(fallback)
    if isinstance(data, list):
        for entry in data:
            if not isinstance(entry, dict):
                continue
            idx = entry.get("post", entry.get("id", entry.get("index", 0)))
            try:
                idx = int(idx) - 1
            except (TypeError, ValueError):
                continue
            if 0 <= idx < len(hits):
                label = str(entry.get("sentiment", "?")).lower()
                if label not in ("positive", "negative", "neutral"):
                    label = "?"
                try:
                    conf = float(entry.get("confidence", 0.0))
                except (TypeError, ValueError):
                    conf = 0.0
                result[idx] = {"sentiment": label, "confidence": max(0.0, min(1.0, conf))}
    return result


def _build_relevance_prompt(question: str, hits) -> str:
    posts = "\n\n".join(
        f"POST {i+1}: {(m['content'] or '').strip()[:400]}"
        for i, (_, m) in enumerate(hits)
    )
    return (
        "You are judging whether each retrieved POST is relevant to the QUESTION.\n"
        "For EACH numbered post, choose ONE label:\n"
        '  - "relevant"  : directly addresses the question\n'
        '  - "partial"   : tangentially related or mentions the topic\n'
        '  - "off_topic" : not related to the question\n'
        "Reply with ONLY a JSON object in this exact shape — one entry per post, "
        "covering ALL posts:\n"
        '{"results":['
        '{"post":1,"label":"relevant","reason":"short reason"},'
        '{"post":2,"label":"off_topic","reason":"short reason"}'
        "]}\n"
        'The "reason" must be a single short sentence (<=12 words).\n\n'
        f"QUESTION: {question}\n\n"
        f"POSTS TO JUDGE:\n{posts}\n\n"
        "JSON:"
    )


def judge_relevance(question: str, hits) -> list[dict]:
    """Single-LLM-call per-post relevance judgment.

    Returns a list aligned with hits: [{'label': str, 'reason': str}, ...].
    Falls back to '?' / '' on failure or when no LLM is available.
    """
    if not hits:
        return []
    fallback = [{"label": "?", "reason": ""} for _ in hits]
    if not ollama_available():
        return fallback
    try:
        raw = _llm_complete(_build_relevance_prompt(question, hits))
    except Exception as e:
        print(f"  (relevance judgment failed: {e})")
        return fallback
    if not raw:
        return fallback

    import json

    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw.strip(), flags=re.IGNORECASE)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if not match:
            return fallback
        try:
            data = json.loads(match.group())
        except json.JSONDecodeError:
            return fallback

    if isinstance(data, dict):
        for key in ("results", "posts", "judgments", "data"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break

    result = list(fallback)
    if isinstance(data, list):
        for entry in data:
            if not isinstance(entry, dict):
                continue
            idx = entry.get("post", entry.get("id", entry.get("index", 0)))
            try:
                idx = int(idx) - 1
            except (TypeError, ValueError):
                continue
            if 0 <= idx < len(hits):
                label = str(entry.get("label", "?")).lower().strip()
                if label not in ("relevant", "partial", "off_topic"):
                    label = "?"
                reason = str(entry.get("reason", "")).strip()
                result[idx] = {"label": label, "reason": reason}
    return result


def evaluate_retrieval(question: str, hits) -> dict:
    """Compute score statistics, diversity, and a confidence verdict over hits."""
    if not hits:
        return {
            "top": 0.0, "mean": 0.0, "min": 0.0, "gap": 0.0,
            "confidence": "LOW",
            "n_sources": 0, "n_authors": 0,
            "date_min": None, "date_max": None,
            "n_hits": 0,
        }

    scores = [s for s, _ in hits]
    top = max(scores)
    bottom = min(scores)
    mean = statistics.fmean(scores)
    gap = top - bottom

    if top >= 0.55 and gap >= 0.05:
        confidence = "HIGH"
    elif top >= 0.40:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    sources = {m.get("source") for _, m in hits if m.get("source")}
    authors = {m.get("author") for _, m in hits if m.get("author")}
    dates = [m.get("date") for _, m in hits if m.get("date")]

    return {
        "top": top,
        "mean": mean,
        "min": bottom,
        "gap": gap,
        "confidence": confidence,
        "n_sources": len(sources),
        "n_authors": len(authors),
        "date_min": min(dates) if dates else None,
        "date_max": max(dates) if dates else None,
        "n_hits": len(hits),
    }


def print_retrieval_evaluation(evaluation: dict, relevance: list[dict]) -> None:
    print_section_header("Retrieval Evaluation")

    conf = evaluation.get("confidence", "LOW")
    conf_color = {"HIGH": "1;32", "MEDIUM": "1;33", "LOW": "1;31"}.get(conf, "1;37")
    conf_str = _c(f"{conf:<10}", conf_color)
    stats = (
        f"(top={evaluation['top']:.2f}, mean={evaluation['mean']:.2f}, "
        f"min={evaluation['min']:.2f}, gap={evaluation['gap']:.2f})"
    )
    print(f"  Confidence:   {conf_str} {_c(stats, '2')}")

    dmin = evaluation.get("date_min")
    dmax = evaluation.get("date_max")
    span = (
        f"{dmin.strftime('%Y-%m-%d')} \u2192 {dmax.strftime('%Y-%m-%d')}"
        if dmin and dmax else "?"
    )
    diversity = (
        f"{evaluation['n_sources']} sources, "
        f"{evaluation['n_authors']} authors, {span}"
    )
    print(f"  Diversity:    {diversity}")

    label_color = {
        "relevant": "1;32",
        "partial":  "1;33",
        "off_topic": "1;31",
        "?":        "2",
    }
    if relevance:
        labels = [r.get("label", "?") for r in relevance]
        if all(lbl == "?" for lbl in labels):
            print(f"  Relevance:    {_c('skipped \u2014 no LLM', '2')}")
        else:
            label_width = max(len(lbl) for lbl in labels)
            label_width = max(label_width, len("off_topic"))
            prefix = "  Relevance:    "
            cont = " " * len(prefix)
            for i, r in enumerate(relevance, 1):
                lbl = r.get("label", "?")
                reason = r.get("reason", "") or ""
                tag = _c(f"[{i}]", "1;33")
                colored_label = _c(f"{lbl:<{label_width}}", label_color.get(lbl, "2"))
                line_prefix = prefix if i == 1 else cont
                sep = _c("\u2014", "2")
                if reason:
                    print(f"{line_prefix}{tag} {colored_label}  {sep} {reason}")
                else:
                    print(f"{line_prefix}{tag} {colored_label}")
    else:
        print(f"  Relevance:    {_c('skipped \u2014 no hits', '2')}")

    warnings: list[str] = []
    if evaluation["n_hits"] > 0 and evaluation["top"] < 0.40:
        warnings.append("Top match is weak; the answer may be unreliable.")
    if evaluation["n_hits"] > 1 and (
        evaluation["n_sources"] <= 1 or evaluation["n_authors"] <= 1
    ):
        warnings.append("Low source diversity.")
    judged = [r.get("label", "?") for r in relevance if r.get("label", "?") != "?"]
    if judged and sum(1 for lbl in judged if lbl == "off_topic") > len(judged) / 2:
        warnings.append("Most retrieved posts judged off-topic.")

    for w in warnings:
        print(f"  {_c('! Warning:', '1;31')}   {_c(w, '1;31')}")


def format_sources_with_sentiment(hits, sentiments) -> str:
    parts = []
    for rank, ((score, m), s) in enumerate(zip(hits, sentiments), 1):
        date = m["date"].strftime("%Y-%m-%d") if m["date"] else "?"
        head = (
            f"[{rank}] {m['source']} #{m['source_id']} | {m['author'] or '?'} | {date}"
            f" | score={score:.3f}"
            f" | sentiment={s['sentiment']} (few-shot: {s['confidence']:.2f})"
        )
        parts.append(head + "\n" + (m["content"] or "").strip())
    return "\n\n".join(parts)


def answer_with_ollama(question: str, hits, temperature: float = TEMPERATURE) -> str:
    import ollama

    client = ollama.Client(host=OLLAMA_HOST)
    user = f"Posts:\n\n{format_context(hits)}\n\nQuestion: {question}"
    resp = client.chat(
        model=OLLAMA_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
        ],
        options={"temperature": temperature},
    )
    return resp["message"]["content"]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ask a RAG question against the ingested social-media chunks.",
        # We accept free-form question words after the flags; argparse handles
        # the flag(s) and we re-join the remainder, so callers can still write
        #   python query.py "what are people saying about X"
        # or
        #   python query.py --temperature 0.2 what are people saying about X
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help=(
            "Sampling temperature for the answer LLM (0.0 = deterministic, "
            "0.7 = balanced default, >=1.0 = more creative). "
            "Overrides the TEMPERATURE env var / config.py default for this run."
        ),
    )
    parser.add_argument(
        "question",
        nargs=argparse.REMAINDER,
        help="The question to ask. If omitted, runs in interactive mode.",
    )
    args = parser.parse_args()

    temperature = args.temperature if args.temperature is not None else TEMPERATURE

    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)

    maybe_auto_ingest(model)

    print(f"Loading vector index from MongoDB ({MONGO_URI}, db={MONGO_DB})...")
    client = MongoClient(MONGO_URI)
    try:
        chunks = client[MONGO_DB][MONGO_CHUNKS_COLLECTION]
        meta, mat = load_index(chunks)
    finally:
        client.close()
    if not meta:
        print("No chunks found. Run `python ingest.py` first.")
        return 1
    print(f"Loaded {len(meta)} chunks ({mat.shape[1]} dims).\n")
    print(f"Generation temperature: {temperature}\n")

    has_llm = ollama_available()
    if not has_llm:
        print(
            f"Warning: Ollama not reachable at {OLLAMA_HOST}. "
            f"Run `ollama serve` and `ollama pull {OLLAMA_MODEL}` — "
            "answers, relevance, and sentiment will be skipped.\n"
        )

    if args.question:
        questions = [" ".join(args.question)]
        interactive = False
    else:
        questions = []
        interactive = True

    while True:
        if interactive:
            try:
                q = input("Question (empty to quit): ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not q:
                break
        else:
            if not questions:
                break
            q = questions.pop(0)
            print(f"Question: {q}")

        hits = retrieve(q, model, meta, mat)

        evaluation = evaluate_retrieval(q, hits)
        try:
            relevance = (
                judge_relevance(q, hits)
                if has_llm
                else [{"label": "?", "reason": ""} for _ in hits]
            )
        except Exception as e:
            print(f"  (relevance judgment failed: {e})")
            relevance = [{"label": "?", "reason": ""} for _ in hits]

        sentiments = (
            classify_sentiments(hits)
            if has_llm
            else [{"sentiment": "?", "confidence": 0.0} for _ in hits]
        )

        print_retrieval_evaluation(evaluation, relevance)

        if has_llm:
            provider = f"Ollama/{OLLAMA_MODEL}"
            answer = answer_with_ollama(q, hits, temperature=temperature)
            print_section_header(f"Answer ({provider})")
            pretty_print_answer(answer)
            print_section_header("Sources (the posts the model was given)")
            pretty_print_sources(hits, sentiments)
        else:
            print_section_header("Ollama unavailable — showing top-K chunks only")
            pretty_print_sources(hits, sentiments)
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
