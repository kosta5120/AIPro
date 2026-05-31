# AI Course Textbook RAG Pipeline

Retrieval-Augmented Generation over the **AI Developers Course Textbook** (PDF).
The pipeline loads and chunks the PDF, embeds sections into **Chroma**, retrieves
with hybrid dense + BM25 search and cross-encoder reranking, and generates
grounded answers via **Ollama** (`gemma3:4b`).

## Project layout

```
data/
  raw/                       # PDF source (AI_Developers_Course_Textbook-v2.pdf)
  processed/                 # Chroma index + chunks.jsonl
  MANIFEST.md                # corpus description
src/
  build_index.py             # build (or rebuild) the Chroma index
  utils.py                   # PDF loading, cleaning, chunking
  retrieval.py               # dense / BM25 / hybrid + reranker
  generation.py              # Ollama generation + citation parsing
  rag_system.py              # exposes `answer(question) -> dict`
  rag_sysytem.py             # compatibility shim (typo filename)
eval/
  build_gold_set.py          # LLM-generate one Q+A per indexed chunk
  gold_set.jsonl
  run_eval.py                # Hit@k, MRR@k, ablation, sample answers
  rag_runs/                  # Markdown reports from rag_system CLI
build_report.py              # generate report.pdf
COMMANDS.md                  # command reference
requirements.txt
```

## Required interface

```python
from src.rag_system import answer

answer("What is retrieval-augmented generation?")
# -> {"answer": "...", "sources": ["pdf_..._chunk_0"], "retrieved_chunks": [...]}
```

`sources` lists `chunk_id` values cited in the LLM reply (`[chunk_id]` markers).
If none are parsed, all retrieved chunk IDs are returned.

Production retrieval in `rag_system.py` uses **hybrid dense + BM25** with a
**cross-encoder reranker** (`BAAI/bge-reranker-large`), then Ollama for generation.

## Install

```powershell
pip install -r requirements.txt
ollama pull gemma3:4b
ollama serve   # if not already running
```

Retrieval works without Ollama; `answer()` returns retrieved chunks and a
generation-unavailable message when the LLM is down.

## Build the index

```powershell
# Default: structured PDF sections, recursive chunking, bge-large embeddings
python src/build_index.py

# Rebuild from scratch
python src/build_index.py --clean

# Smoke test (first 30 pages per PDF)
python src/build_index.py --max-per-source 30 --clean
```

Key flags: `--load-mode {full,page,structured}`, `--chunk-strategy {fixed,sentence,recursive}`,
`--chunk-size`, `--overlap`, `--persist-dir`, `--clean`.

See [`COMMANDS.md`](COMMANDS.md) for the full flag list.

## Ask a question

```powershell
python src/rag_system.py "What is the attention mechanism?"
python src/rag_system.py --json "What does RAG stand for?"
```

Reports are written under `eval/rag_runs/` by default.

## Evaluate

```powershell
python eval/build_gold_set.py --limit 20   # smoke test
python eval/build_gold_set.py              # full gold set (~one LLM call per chunk)
python eval/run_eval.py                    # ablation: dense k=3/5/8, BM25, hybrid
```

`run_eval.py` writes `eval/results.md` and samples answers to
`eval/answers_sample.jsonl`. Metrics: **Hit@k**, **MRR@k**, and **answer accuracy**
(token F1 vs reference answers).

## Corpus

Single PDF textbook (152 pages, 12 modules). Details in [`data/MANIFEST.md`](data/MANIFEST.md).

Embeddings: `BAAI/bge-large-en-v1.5` (Chroma, cosine).  
LLM: `gemma3:4b` via Ollama (local only, no API keys).

## Configuration

The PDF pipeline reads settings from **environment variables** in
`src/generation.py` and `src/rag_system.py` (not from a root `config.py`):

| Variable | Default | Effect |
|---|---|---|
| `RAG_TOP_K` | reranker default | Chunks passed to the LLM after reranking |
| `RAG_LLM_MODEL` | `gemma3:4b` | Ollama model for generation |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |

## Notes

* Reproducible index: delete `data/processed/chroma` and rerun `build_index.py --clean`.
* `rag_sysytem.py` re-exports `rag_system` for the common filename typo.
* Mid-term PDF report: `python build_report.py` → `report.pdf`.
