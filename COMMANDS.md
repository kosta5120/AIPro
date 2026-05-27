# Project Commands Reference

## Setup

```bash
# Install all Python dependencies
pip install -r requirements.txt

# Pull the LLM (only needed once)
ollama pull gemma3:4b

# Start the Ollama server (if not already running)
ollama serve
```

---

## 1. Build the vector index — `src/build_index.py`

```bash
# Default (fixed chunking, 500 words, 80 overlap, all PDF pages)
python src/build_index.py

# Custom chunk size / overlap
python src/build_index.py --chunk-size 300 --overlap 60

# Sentence-based chunking
python src/build_index.py --chunk-strategy sentence --chunk-size 400

# Limit pages per PDF (fast smoke test)
python src/build_index.py --max-per-source 30 --clean

# Rebuild into a named collection + custom path
python src/build_index.py --collection-name pdf_docs --persist-dir data/processed/chroma --clean

# Full flag reference:
#   --raw-dir          data/raw            source folder for PDFs
#   --chunk-strategy   fixed | sentence
#   --chunk-size       int (words)
#   --overlap          int (words)
#   --max-per-source   int (pages per PDF)
#   --collection-name  str
#   --persist-dir      str
#   --chunks-path      str  (where to save chunks.jsonl)
#   --batch-size       int  (embedding batch size, default 128)
#   --clean            delete persist-dir before building
```

---

## 2. Ask a question — `src/rag_system.py`

```bash
# Quick CLI question
python src/rag_system.py "What is the main topic of this document?"

# Or import programmatically
python -c "from src.rag_system import answer; import json; print(json.dumps(answer('What is RAG?'), indent=2))"
```

**Environment variables** that affect this command:

| Variable | Default | Effect |
|---|---|---|
| `RAG_TOP_K` | `5` | Number of chunks retrieved |
| `RAG_LLM_MODEL` | `gemma3:4b` | Ollama model used for generation |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |

---

## 3. Build the gold evaluation set — `eval/build_gold_set.py`

```bash
# Smoke test (only first 20 chunks)
python eval/build_gold_set.py --limit 20

# Full run (one LLM call per chunk — can be long)
python eval/build_gold_set.py

# Resume an interrupted run
python eval/build_gold_set.py --resume

# Start from chunk 1000, process 500
python eval/build_gold_set.py --start 1000 --limit 500

# Full flag reference:
#   --chunks-path   data/processed/chunks.jsonl   input chunks
#   --out           eval/gold_set.jsonl            output file
#   --model         gemma3:4b                      Ollama model
#   --ollama-host   http://localhost:11434
#   --limit         int   max chunks to process
#   --start         int   skip first N chunks
#   --resume              append, skip already-done chunks
#   --temperature   0.2
#   --seed          42    category rotation seed
#   --timeout       120.0 seconds per LLM call
```

---

## 4. Run the ablation evaluation — `eval/run_eval.py`

```bash
python eval/run_eval.py
```

Builds 2 indices (chunk_size 300 and 700), evaluates 4 configs (k = 3 / 5 / 8),
computes **Hit@k**, **MRR@k**, and **Answer accuracy** (token F1), and writes
results to `eval/results.md` and `eval/answers_sample.jsonl`.

---

## 5. Generate the PDF report — `build_report.py`

```bash
python build_report.py
# → writes report.pdf in the project root
```

> Requires `reportlab`: `pip install reportlab`

---

## Typical end-to-end workflow

```bash
pip install -r requirements.txt           # 1. install deps
ollama pull gemma3:4b && ollama serve     # 2. start LLM
python src/build_index.py --clean         # 3. build index from PDF
python eval/build_gold_set.py --limit 50  # 4. build gold set (smoke test)
python eval/run_eval.py                   # 5. run ablation + metrics
python src/rag_system.py "What is a transformer?"  # 6. ask a question
python build_report.py                    # 7. generate PDF report
```
