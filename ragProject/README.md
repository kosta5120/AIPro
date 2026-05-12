# Social-Media RAG over MongoDB

A Retrieval-Augmented Generation system over the `FacebookPosts`,
`InstagramPosts`, and `TwitterPosts` collections in a local MongoDB.
Ask natural-language questions in any language; the system returns an
LLM-written answer grounded in the actual posts, with citations.

---

## What this project does

1. Reads social-media posts from three MongoDB collections.
2. Splits each post into chunks and embeds each chunk to a 384-dimensional
   vector using a free, local multilingual model.
3. Stores the chunks **and** their vectors in the MongoDB `rag_chunks`
   collection in the same database.
4. At query time:
   - Embeds your question with the same model.
   - Computes cosine similarity against every chunk in the index.
   - Returns the top-K most relevant chunks.
   - Sends those chunks plus your question to an LLM (local Gemma 3 by default).
   - Classifies each retrieved chunk's sentiment using few-shot prompting.
5. Prints the LLM's answer plus the sources it was based on, with each source
   labelled by relevance score and sentiment.

No data ever leaves your machine when running with the local Gemma 3 model.

---

## Pipeline diagrams

### Indexing (run once + whenever new posts arrive)

```
+----------------+     +----------+     +--------------+     +---------------+
| FacebookPosts  |     |          |     | Multilingual |     |               |
| InstagramPosts | --> |  chunks  | --> |   MiniLM     | --> |  rag_chunks   |
| TwitterPosts   |     | (<=800 ch)|    |  (384-dim)   |     |  (MongoDB)    |
+----------------+     +----------+     +--------------+     +---------------+
   Data Loading        Splitting         Embedding             Vector store
                                                                + content_hash
                                                                  for dedupe
```

### Retrieval & Generation (every query)

```
   "Show posts about trump"
            |
            | encode (same MiniLM)
            v
   [384-dim query vector]
            |
            | matmul against (N, 384) matrix
            | (cosine since both are unit-normalized)
            v
   top-5 highest-scoring chunks  --+
                                   |
                                   v
                          +------------------+
   original question  --> |       LLM        | --> answer with [1], [2] citations
                          | (Gemma3 / Gemini |
                          |   / Claude)      |
                          +------------------+

   In parallel: classify_sentiments(top-5)
        few-shot prompt: 9 labelled examples + 5 retrieved posts
        --> JSON: [{post:1, sentiment:'neutral', confidence:0.85}, ...]
```

---

## File map

| File | Role |
|---|---|
| `setup_db.py` | Verifies the Mongo connection and creates the `rag_chunks` indexes. Run once. |
| `config.py` | Central config: Mongo URI / database / collections, model names, chunk sizes, provider routing flags. Reads `.env`. |
| `ingest.py` | The indexing pipeline. Reads source collections → chunks → embeds → inserts into `rag_chunks`. Idempotent + dedup-aware. |
| `dedupe.py` | One-shot cleanup that backfills `content_hash` on existing chunks and removes duplicates. |
| `query.py` | The retrieval + generation pipeline. Run for every question. |
| `_inspect.py` | Utility to print counts + a sample document for each source collection. |
| `requirements.txt` | Python dependencies. |
| `.env` | Local secrets / flags (Mongo URI, Ollama, optional API keys). |

---

## Setup

### Prerequisites

- **Python 3.11+**
- **MongoDB** running locally (no auth) on `mongodb://localhost:27017`
- **Ollama** (for the default local LLM): <https://ollama.com/download/windows>

### One-time install

```powershell
cd C:\Users\kosta\Desktop\ragProject
pip install -r requirements.txt
ollama pull gemma3:4b
python setup_db.py
```

### Configure `.env`

```
MONGO_URI=mongodb://localhost:27017
MONGO_DB=admin

USE_OLLAMA=1                 # use local Gemma 3 (default - no API key needed)
OLLAMA_MODEL=gemma3:4b       # or gemma3:12b, qwen2.5:7b, etc.
OLLAMA_HOST=http://localhost:11434

TEMPERATURE=0.7              # 0.0 = deterministic, 0.7 = balanced, >=1.0 = creative

# optional:
# GEMINI_API_KEY=AIza...
# ANTHROPIC_API_KEY=sk-ant-...
```

You can override the source collections without editing code:

```
MONGO_COLLECTIONS=FacebookPosts,InstagramPosts,TwitterPosts
MONGO_CHUNKS_COLLECTION=rag_chunks
```

**Provider precedence at query time:** Ollama (if `USE_OLLAMA=1`) → Gemini (if key set) → Claude (if key set) → no-LLM fallback (prints chunks only).

---

## How to use

### 1. Verify the connection

```powershell
python _inspect.py
```

Prints counts and a sample document from each source collection.

### 2. Build the index (one time)

```powershell
python ingest.py
```

What it does:

- Connects to MongoDB with the URI from `config.py` / `.env`.
- Ensures the `rag_chunks` unique + hash indexes exist.
- Loads existing content hashes for deduplication.
- For each source collection (`FacebookPosts`, `InstagramPosts`, `TwitterPosts`):
  - Reads all documents.
  - Concatenates the text fields (`UpperPost + DownPost` for Facebook,
    `Text` for Instagram, `Text` for Twitter).
  - Splits each post into chunks of ≤800 characters with 100-char overlap.
  - Computes SHA-256 of each chunk's UTF-8 bytes — skips chunks already
    present by `(source, source_id, chunk_index)` AND skips byte-identical
    duplicates by hash.
  - Embeds each new chunk with `paraphrase-multilingual-MiniLM-L12-v2`
    (384 dims, normalized).
  - Inserts in batches of 64.

Re-runnable: subsequent runs only insert new chunks.

### 3. Clean up duplicates from existing data (optional)

```powershell
python dedupe.py
```

Backfills `content_hash` for any chunks that lack it and deletes
duplicates, keeping the lowest `_id` per hash. Safe to re-run.

### 4. Ask questions

**One-shot:**
```powershell
python query.py "Show posts about trump"
python query.py "מי פרסם על האירוע אתמול"
python query.py "what are people saying about cybersecurity"
```

**Interactive:**
```powershell
python query.py
```

**Override the sampling temperature for a single run:**
```powershell
python query.py --temperature 0.0 "Summarize the posts about election fraud"   # factual, deterministic
python query.py --temperature 0.7 "what are people saying about cybersecurity" # balanced (default)
python query.py --temperature 1.2 "Write a creative recap of recent posts"     # more diverse output
```

The `--temperature` flag takes precedence over the `TEMPERATURE` env var /
`config.py` default for that run only. It is forwarded to whichever provider
ends up answering (Ollama / Gemini / Claude). A lower value (closer to 0)
makes the next-token distribution sharper, so the model picks the highest-
probability token more often — best for factual RAG answers grounded in the
retrieved posts. Higher values flatten the distribution and produce more
varied, creative phrasings.

#### Auto-ingest on query

Every `python query.py ...` run first does a cheap check against MongoDB:
for each collection in `MONGO_COLLECTIONS` it compares the source document
count to the number of distinct `source_id` values already present in
`rag_chunks` for that platform. If at least one platform has new posts, the
ingest pipeline runs automatically (reusing the already-loaded embedding
model) before the query is answered; otherwise the check costs one
`estimated_document_count` + one indexed `distinct` per platform and the
query proceeds immediately.

To keep the check fast on repeat runs, a tiny `_rag_meta` collection caches
the last-seen source-document count per platform. The cache is updated
every time `query.py` runs the check (whether or not new chunks were
actually written), so platforms whose source posts produce zero chunks
(empty/duplicate text) don't cause the auto-ingest to re-trigger forever.

To disable auto-ingest entirely and require manual `python ingest.py` runs,
set the env var below:

```
AUTO_INGEST=0
```

> **Note:** if you ever drop, truncate, or wipe `rag_chunks` manually, also
> drop `_rag_meta` (`db._rag_meta.drop()` in `mongosh`) — otherwise the
> cache will falsely report "up to date" and your next query will return
> nothing until you re-run `python ingest.py` by hand.

---

## Anatomy of a query

When you run `python query.py "your question"`:

1. **Load the embedding model** (~120 MB on first run, then cached locally).
2. **Load the vector index from MongoDB**:
   - Reads every document from the `rag_chunks` collection.
   - All embeddings are stacked into one NumPy matrix `mat` of shape `(N, 384)`.
   - Metadata stays in a parallel list `meta`.
3. **Embed the question** with the same multilingual MiniLM model.
4. **Cosine similarity** in one matmul:
   ```python
   scores = mat @ q          # shape (N,)
   top_idx = np.argsort(-scores)[:K]
   ```
   Both `mat` rows and `q` are L2-normalized at encode time, so dot product
   equals cosine similarity. K defaults to 5.
5. **Pick the LLM provider** based on `.env`:
   - Ollama (Gemma 3) if `USE_OLLAMA=1`
   - Gemini if `GEMINI_API_KEY` set
   - Claude if `ANTHROPIC_API_KEY` set
   - Otherwise: print chunks only, no generation.
6. **Generate the answer**: send a system prompt + the K retrieved posts + the
   original question to the LLM. Response includes `[1]`, `[2]` citations.
7. **Few-shot sentiment classification**: in a single additional LLM call,
   classify all K retrieved posts as positive / negative / neutral with a
   self-reported confidence in `[0, 1]`. The prompt includes 9 multilingual
   labelled examples (3 per class) before the posts to classify.
8. **Print** the answer, then the sources block:
   ```
   [1] twitter #129985 | TruthTrumpPost | 2026-05-03 | score=0.708 | sentiment=neutral (few-shot: 0.85)
   WATCH: Secretary Rubio in action as a wedding DJ ...
   ```

Each source line shows:
- `score=` cosine similarity to your question (retrieval relevance, 0–1)
- `sentiment=` few-shot label
- `few-shot:` the model's confidence in that label (0–1)

---

## Configuration knobs (`config.py` / `.env`)

| Setting | Default | Effect |
|---|---|---|
| `MONGO_URI` | `mongodb://localhost:27017` | Where to find MongoDB. |
| `MONGO_DB` | `admin` | Database containing the source + `rag_chunks` collections. |
| `MONGO_COLLECTIONS` | `FacebookPosts,InstagramPosts,TwitterPosts` | Comma-separated source collections to ingest. |
| `MONGO_CHUNKS_COLLECTION` | `rag_chunks` | Where the embedded chunks are written. |
| `AUTO_INGEST` | `1` | When truthy, `query.py` checks for new posts in MongoDB and auto-runs ingest before answering. Set to `0` to skip. |
| `EMBEDDING_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Used at ingest AND query — must match. Changing requires re-running `ingest.py` from scratch. |
| `EMBEDDING_DIM` | 384 | Must match the model. |
| `CHUNK_SIZE` | 800 | Max characters per chunk. |
| `CHUNK_OVERLAP` | 100 | Overlap between consecutive chunks. |
| `TOP_K` | 5 | How many chunks to retrieve per query. |
| `USE_OLLAMA` | true (when set in `.env`) | Use local Ollama as the LLM. |
| `OLLAMA_MODEL` | `gemma3:4b` | Any Ollama-pulled model. |
| `LLM_MODEL` | `claude-haiku-4-5` | Used only when falling back to Claude. |
| `GEMINI_MODEL` | `gemini-2.5-flash` | Used only when falling back to Gemini. |
| `TEMPERATURE` | `0.7` | Sampling temperature for the answer LLM. `0.0` = deterministic / greedy decoding (sharpest next-token distribution — best for factual RAG answers), `0.7` = balanced default, `>=1.0` = more creative / diverse output. Override per run with `python query.py --temperature 0.2 "..."`. Applied uniformly to Claude, Gemini, and Ollama. |

---

## `rag_chunks` document shape

Each chunk written by `ingest.py` looks like:

| Field | Type | Purpose |
|---|---|---|
| `_id` | `ObjectId` | Auto-generated chunk id. |
| `source` | `str` | Platform tag: `facebook` / `instagram` / `twitter`. |
| `source_collection` | `str` | Name of the source Mongo collection. |
| `source_id` | `str` | Original document's `_id`, stringified. |
| `natural_id` | `str` \| `null` | Platform-natural id (`FacebookId` / `InstagramId` / `TwitterId`). |
| `chunk_index` | `int` | Position within the post (0 if it fits in one chunk). |
| `author` | `str` \| `null` | Display name / username. |
| `created_date` | `datetime` \| `null` | Original post timestamp. |
| `content` | `str` | Chunk text. |
| `content_hash` | `Binary` (32 bytes) | SHA-256 of UTF-8 bytes of `content` — used for deduplication. |
| `embedding` | `Binary` | Float32 384-dim vector, raw bytes (`numpy.tobytes()`). |
| `ingested_at` | `datetime` | Insertion timestamp (UTC). |

Indexes:
- Unique compound on `(source, source_id, chunk_index)` — prevents re-inserting the same chunk on re-runs.
- Non-unique on `content_hash` for fast dedupe lookup.

### Auxiliary `_rag_meta` collection

Created automatically the first time `query.py` runs the auto-ingest check.
One small document per platform, used only as a high-water-mark cache so
the cheap "are there new posts?" check can short-circuit:

```json
{ "platform": "twitter", "last_seen_source_count": 3966 }
```

Safe to drop at any time — it will be rebuilt on the next query.

MongoDB Community 7+ has no native vector type (Atlas Vector Search lives
in the cloud), so similarity is computed in Python with a NumPy matmul
against the in-memory matrix. Fine up to ~500k chunks.

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `ServerSelectionTimeoutError` | MongoDB not running | Start the `MongoDB` service (`services.msc` on Windows). |
| `No chunks found. Run ingest.py first.` | Index is empty | `python ingest.py` (or just run any `python query.py ...` — auto-ingest will populate it). |
| `query.py` says "up to date" but returns no results | `rag_chunks` was wiped but `_rag_meta` cache is stale | Drop the cache: `mongosh admin --eval "db._rag_meta.drop()"`, then re-query. |
| Auto-ingest runs on every query even though no new posts arrived | `_rag_meta` missing or `AUTO_INGEST` env var unset on a fresh DB | First query rebuilds the cache; subsequent queries will short-circuit. |
| `cp1252 codec can't encode character` | Windows console can't print non-ASCII | Already fixed — `query.py` reconfigures stdout/stderr to UTF-8 at import time. |
| `Your credit balance is too low` | Anthropic key has $0 credits | Use Gemini (free tier) or Ollama (free, local). |
| `(no LLM key set — showing top-K chunks only)` | No provider configured | Either set `USE_OLLAMA=1` (and have Ollama running) or add `GEMINI_API_KEY` / `ANTHROPIC_API_KEY` to `.env`. |
| Sentiments all show `?` and confidence `0.00` | LLM returned malformed JSON | The parser tolerates several formats. If it persists, run `python -c "from query import _build_sentiment_prompt, _llm_complete; ..."` to inspect the raw response. |

---

## Tuning quality

| Want | Action |
|---|---|
| Smarter answers | `ollama pull gemma3:12b` then `OLLAMA_MODEL=gemma3:12b` in `.env` (~10 GB RAM/VRAM). |
| Better retrieval (more recall) | Increase `TOP_K` in `config.py`. |
| Better retrieval (cross-language quality) | Try a stronger multilingual embedder, e.g. `intfloat/multilingual-e5-large` (1024 dims) — change `EMBEDDING_MODEL` and `EMBEDDING_DIM`, then drop and re-build `rag_chunks`. |
| Better sentiment labels | Add more diverse labelled examples to `FEW_SHOT_SENTIMENT_EXAMPLES` in `query.py`, especially in the languages your posts use. 5–15 examples is the sweet spot. |
| Faster queries | Skip sentiment classification (saves one LLM call), or move retrieval to FAISS for sub-millisecond ANN. |

---

## What this project deliberately does *not* do

- **No ANN index.** Brute-force NumPy matmul is sufficient for ~10k chunks.
- **No Atlas Vector Search.** The local MongoDB Community Edition has no
  native vector type; we compute similarity in Python.
- **No re-ranker.** A cross-encoder pass would improve the top-K ordering at
  the cost of a second model load.
- **No conversation memory.** Each query is independent.
- **No metadata filters.** "Only Facebook posts after 2025-01-01" would be a
  Python filter on `meta` / `mat` before the matmul — easy to add.
- **No streaming output.** Answers are returned as one block.
