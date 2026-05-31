"""Central config for the RAG project."""
import os
from dotenv import load_dotenv

load_dotenv()


#CONN_STR = (
#   "Driver={ODBC Driver 18 for SQL Server}
#    "Server=.\\sqlexpress;"
#    "Database=CyberglobesExportDB;"
#    "Trusted_Connection=yes;"
#    "Encrypt=no;"
#    "TrustServerCertificate=yes;"
#)
# MongoDB — source posts live in the collections listed below; the embedded
# chunks + vectors are written to MONGO_CHUNKS_COLLECTION in the same DB.
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB = os.getenv("MONGO_DB", "admin")
MONGO_COLLECTIONS = [
    c.strip()
    for c in os.getenv(
        "MONGO_COLLECTIONS",
        "FacebookPosts,InstagramPosts,TwitterPosts",
    ).split(",")
    if c.strip()
]
MONGO_CHUNKS_COLLECTION = os.getenv("MONGO_CHUNKS_COLLECTION", "rag_chunks")

# Multilingual model — handles Hebrew/Arabic/English posts. 384 dims.
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_DIM = 384

# Chunking — most social posts are short, but FB posts can be long.
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Retrieval
TOP_K = 5

# Generation — local LLM via Ollama (gemma3:4b by default).
# TEMPERATURE controls the next-token probability distribution at generation
# time. 0.0 = deterministic / greedy decoding (best for factual RAG answers),
# ~0.7 = balanced default, >=1.0 = noticeably more creative / random. Can be
# overridden per-run with the `--temperature` CLI flag on `query.py`.
def _parse_temperature(raw: str | None, default: float = 0.7) -> float:
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


TEMPERATURE = _parse_temperature(os.getenv("TEMPERATURE"))

OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Auto-ingest: every `python query.py ...` run will first check MongoDB for
# new source posts and ingest them before answering. Defaults to True; set
# AUTO_INGEST=0 in .env (or the shell) to skip the check entirely.
AUTO_INGEST = os.getenv("AUTO_INGEST", "1").lower() in ("1", "true", "yes")
