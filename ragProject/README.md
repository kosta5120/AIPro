# Social-Media RAG over MongoDB

A Retrieval-Augmented Generation system over the `FacebookPosts`,
`InstagramPosts`, and `TwitterPosts` collections in a local MongoDB.
Ask natural-language questions in any language; the system returns an
LLM-written answer grounded in the actual posts, with citations, plus
retrieval diagnostics and per-post sentiment labels.

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
   - Scores retrieval confidence (HIGH / MEDIUM / LOW) from similarity stats.
   - Optionally asks the LLM to label each hit as `relevant`, `partial`, or
     `off_topic` for the question.
   - Classifies each retrieved chunk's sentiment using few-shot prompting.
   - Sends the top-K chunks plus your question to **Ollama** (`gemma3:4b`)
     and prints the answer with `[1]`, `[2]` citations.
5. Prints structured sections: retrieval evaluation, answer, then sources with
   relevance score and sentiment per post.

All generation runs locally via Ollama — no cloud API keys required.

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

### Retrieval & generation (every query)

```
   "Show posts about trump"
            |
            | encode (same MiniLM)
            v
   [384-dim query vector]
            |
            | matmul against (N, 384) matrix
            | (cosine — rows and query are L2-normalized)
            v
   top-K highest-scoring chunks
            |
            +--> evaluate_retrieval()  --> confidence HIGH/MEDIUM/LOW, diversity
            |
            +--> judge_relevance()     --> relevant | partial | off_topic (LLM)
            |
            +--> classify_sentiments() --> positive | negative | neutral (LLM)
            |
            v
   +------------------+     pretty-printed answer + sources
   | Ollama gemma3:4b | --> with [1], [2] citations
   +------------------+
```

---

## File map

| File | Role |
|---|---|
| `setup_db.py` | Verifies the Mongo connection and creates the `rag_chunks` indexes. Run once. |
| `config.py` | Central config: Mongo URI / database / collections, model names, chunk sizes, provider flags. Reads `.env`. |
| `ingest.py` | Indexing pipeline. Reads source collections → chunks → embeds → inserts into `rag_chunks`. Idempotent + dedup-aware. |
| `dedupe.py` | One-shot cleanup that backfills `content_hash` on existing chunks and removes duplicates. |
| `query.py` | Retrieval + evaluation + generation pipeline. Run for every question. |
| `few_shot_qa.py` | Library of 50 fictional analyst Q&A examples (`FEW_SHOT_QA_EXAMPLES`) and `format_few_shot_block()` for prompt injection experiments. Not wired into `query.py` by default. |
| `_inspect.py` | Utility to print counts + a sample document for each source collection. |
| `requirements.txt` | Python dependencies. |
| `.env` | Local flags (Mongo URI, Ollama host/model, temperature). |

---

## Setup

### Prerequisites

- **Python 3.11+**
- **MongoDB** running locally (no auth) on `mongodb://localhost:27017`
- **Ollama** with `gemma3:4b` pulled: <https://ollama.com/download/windows>

### One-time install

```powershell
cd {location of the project}
pip install -r requirements.txt
ollama pull gemma3:4b
python setup_db.py
```

### Configure `.env`

```
MONGO_URI=mongodb://localhost:27017
MONGO_DB=admin

OLLAMA_MODEL=gemma3:4b
OLLAMA_HOST=http://localhost:11434

TEMPERATURE=0.7              # 0.0 = deterministic, 0.7 = balanced, >=1.0 = creative
```

Override source collections without editing code:

```
MONGO_COLLECTIONS=FacebookPosts,InstagramPosts,TwitterPosts
MONGO_CHUNKS_COLLECTION=rag_chunks
```

**LLM:** Every generation, relevance, and sentiment call goes to Ollama
(`OLLAMA_MODEL`, default `gemma3:4b`). If Ollama is not running, `query.py`
prints retrieved chunks only and skips those LLM steps.

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
  - Concatenates the text fields (`UpperPost` + `DownPost` for Facebook,
    `Text` for Instagram and Twitter).
  - Splits each post into chunks of ≤800 characters with 100-char overlap.
  - Computes SHA-256 of each chunk's UTF-8 bytes — skips chunks already
    present by `(source, source_id, chunk_index)` AND skips byte-identical
    duplicates by hash.
  - Embeds each new chunk with
    `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`
    (384 dims, L2-normalized).
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

**Override temperature for a single run:**

```powershell
python query.py --temperature 0.0 "Summarize the posts about election fraud"
python query.py --temperature 0.7 "what are people saying about cybersecurity"
python query.py --temperature 1.2 "Write a creative recap of recent posts"
```

The `--temperature` flag overrides `TEMPERATURE` in `.env` / `config.py` for
that run only. Lower values sharpen the next-token distribution (better for
factual RAG); higher values produce more varied phrasing.

#### Auto-ingest on query

Every `python query.py ...` run first checks MongoDB: for each collection in
`MONGO_COLLECTIONS` it compares the source document count to a per-platform
high-water mark in `_rag_meta` (or falls back to `distinct source_id` counts
in `rag_chunks`). If any platform has new posts, ingest runs automatically
before the question is answered.

```
AUTO_INGEST=0
```

disables this check; you must run `python ingest.py` manually.

> **Note:** if you drop or truncate `rag_chunks`, also drop `_rag_meta`
> (`db._rag_meta.drop()` in `mongosh`) — otherwise the cache may report
> "up to date" while the index is empty.

---

## Anatomy of a query

When you run `python query.py "your question"`:

1. **Load the embedding model** (~120 MB on first run, then cached).
2. **Auto-ingest** (unless `AUTO_INGEST=0`) if new source posts exist.
3. **Load the vector index** from `rag_chunks`:
   - All embeddings stacked into a NumPy matrix `mat` of shape `(N, 384)`.
   - Metadata in a parallel list `meta`.
4. **Embed the question** with the same multilingual MiniLM model.
5. **Cosine similarity** in one matmul (`scores = mat @ q`, top-K via
   `argsort`). K defaults to 5 (`TOP_K` in `config.py`).
6. **Retrieval evaluation** — confidence from top/mean/min scores and gap;
   diversity (sources, authors, date span); warnings for weak top score,
   low diversity, or mostly off-topic judgments.
7. **Relevance judgment** (one Ollama call, JSON) — each hit labelled
   `relevant` / `partial` / `off_topic` with a short reason. Skipped if
   Ollama is unavailable.
8. **Sentiment classification** (one Ollama call, JSON) — 9 multilingual
   few-shot examples in `FEW_SHOT_SENTIMENT_EXAMPLES`, then labels for all
   K posts with confidence in `[0, 1]`.
9. **Answer generation** — system prompt + retrieved posts + question to
   Ollama. Response uses `[1]`, `[2]` citations; `pretty_print_answer` splits
   the text into readable blocks.
10. **Sources** — `pretty_print_sources` prints each post with score,
    sentiment, and confidence.

Example source line:

```
[1] twitter #129985 | TruthTrumpPost | 2026-05-03 | score=0.708  sentiment=neutral (0.85)
    WATCH: Secretary Rubio in action as a wedding DJ ...
```

---

## Configuration knobs (`config.py` / `.env`)

| Setting | Default | Effect |
|---|---|---|
| `MONGO_URI` | `mongodb://localhost:27017` | MongoDB connection string. |
| `MONGO_DB` | `admin` | Database for source + `rag_chunks` collections. |
| `MONGO_COLLECTIONS` | `FacebookPosts,InstagramPosts,TwitterPosts` | Comma-separated source collections to ingest. |
| `MONGO_CHUNKS_COLLECTION` | `rag_chunks` | Embedded chunk store. |
| `AUTO_INGEST` | `1` | Auto-run ingest on `query.py` when new posts exist. Set `0` to skip. |
| `EMBEDDING_MODEL` | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` | Must match at ingest and query. Changing requires re-running `ingest.py`. |
| `EMBEDDING_DIM` | `384` | Must match the model. |
| `CHUNK_SIZE` | `800` | Max characters per chunk. |
| `CHUNK_OVERLAP` | `100` | Overlap between consecutive chunks. |
| `TOP_K` | `5` | Chunks retrieved per query. |
| `OLLAMA_MODEL` | `gemma3:4b` | Ollama model for answers, relevance, and sentiment. |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL. |
| `TEMPERATURE` | `0.7` | Sampling temperature. Override per run with `--temperature`. |

---

## `rag_chunks` document shape

Each chunk written by `ingest.py`:

| Field | Type | Purpose |
|---|---|---|
| `_id` | `ObjectId` | Auto-generated chunk id. |
| `source` | `str` | Platform: `facebook` / `instagram` / `twitter`. |
| `source_collection` | `str` | Source Mongo collection name. |
| `source_id` | `str` | Original document `_id`, stringified. |
| `natural_id` | `str` \| `null` | Platform id (`FacebookId` / `InstagramId` / `TwitterId`). |
| `chunk_index` | `int` | Position within the post (0 if single chunk). |
| `author` | `str` \| `null` | Display name / username. |
| `created_date` | `datetime` \| `null` | Original post timestamp. |
| `content` | `str` | Chunk text. |
| `content_hash` | `Binary` (32 bytes) | SHA-256 of UTF-8 `content` for dedupe. |
| `embedding` | `Binary` | Float32 384-dim vector (`numpy.tobytes()`). |
| `ingested_at` | `datetime` | Insertion timestamp (UTC). |

**Indexes:**

- Unique compound on `(source, source_id, chunk_index)`.
- Non-unique on `content_hash`.

### Auxiliary `_rag_meta` collection

Per-platform high-water mark for the auto-ingest check:

```json
{ "platform": "twitter", "last_seen_source_count": 3966 }
```

Safe to drop anytime — rebuilt on the next query.

MongoDB Community 7+ has no native vector type; similarity is computed in
Python with NumPy matmul against the in-memory matrix. Fine up to ~500k
chunks.

---

## Few-shot Q&A library (`few_shot_qa.py`)

`FEW_SHOT_QA_EXAMPLES` contains 50 hand-written analyst-style Q&A pairs
(fictional authors and events) across English, Hebrew, and Arabic — covering
attribution, sentiment, trends, threats, networks, timelines, geography,
disinformation, cross-platform comparison, and "posts don't answer" cases.

Run `python few_shot_qa.py` to print the example count. Use
`format_few_shot_block()` to render them for prompt experiments; the main
`query.py` path does not inject this block yet (only sentiment few-shots are
used in production queries).

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `ServerSelectionTimeoutError` | MongoDB not running | Start the `MongoDB` service (`services.msc` on Windows). |
| `No chunks found. Run ingest.py first.` | Index is empty | `python ingest.py`, or run any `query.py` query with auto-ingest enabled. |
| Query says "up to date" but returns nothing | `rag_chunks` wiped, `_rag_meta` stale | `mongosh admin --eval "db._rag_meta.drop()"`, then re-query or re-ingest. |
| Auto-ingest on every query | Missing `_rag_meta` on fresh DB | First query rebuilds cache; later queries short-circuit. |
| `cp1252 codec can't encode character` | Windows console encoding | Fixed — `query.py` sets stdout/stderr to UTF-8 at import. |
| `Ollama unavailable — showing top-K chunks only` | Ollama not running | `ollama serve` and `ollama pull gemma3:4b`. |
| Relevance / sentiment all `?` | Ollama down or malformed JSON | Check Ollama is up; inspect raw JSON from `_llm_complete`. |
| Sentiments `?` and confidence `0.00` | Parse failure | Same as above; tolerance for several JSON shapes is built in. |

---

## Tuning quality

| Goal | Action |
|---|---|
| Smarter answers (more RAM) | `ollama pull gemma3:12b` then `OLLAMA_MODEL=gemma3:12b` in `.env`. |
| More recall | Increase `TOP_K` in `config.py`. |
| Better cross-language retrieval | Stronger embedder (e.g. `intfloat/multilingual-e5-large`, 1024 dims) — update `EMBEDDING_MODEL` / `EMBEDDING_DIM`, drop `rag_chunks`, re-ingest. |
| Better sentiment labels | Extend `FEW_SHOT_SENTIMENT_EXAMPLES` in `query.py` (5–15 diverse examples per language is a good target). |
| Better answer style | Wire `format_few_shot_block()` from `few_shot_qa.py` into the generation prompt in `query.py`. |
| Faster queries | Disable relevance or sentiment (comment out calls in `main()`), or add FAISS for sub-millisecond ANN. |

---

## What this project deliberately does *not* do

- **No ANN index.** Brute-force NumPy matmul is sufficient for ~10k chunks.
- **No Atlas Vector Search.** Local MongoDB Community has no native vector type.
- **No cross-encoder re-ranker.** Would improve ordering at the cost of another model load.
- **No conversation memory.** Each query is independent.
- **No metadata filters.** Date/platform filters would be a pre-matmul mask on `meta` — easy to add.
- **No streaming output.** Answers are returned as one block.
