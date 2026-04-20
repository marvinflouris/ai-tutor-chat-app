# config.py
# ─────────────────────────────────────────────────────────────────────────────
# Single source of truth for every runtime setting in Gnosis by Zoe.
# To switch LLM or embedder: change LLM_PROVIDER or EMBEDDER_MODE here only.
# Every other module reads from this file — nothing else needs touching.
#
# EMBEDDER_MODE options:
#   "cohere"          → Cohere embed-english-v3.0 (1024-dim) via Cohere API
#   "llama_pinecone"  → Nvidia Llama Text Embed v2 (1024-dim) via Pinecone Inference API
#                       Pinecone generates the embedding — no local model needed
#   "huggingface"     → all-MiniLM-L6-v2 (384-dim) local, no API key, dev only
# ─────────────────────────────────────────────────────────────────────────────

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── PROJECT PATHS ─────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).resolve().parent
STORAGE_DIR   = BASE_DIR / "rag_storage"
LOGS_DIR      = BASE_DIR / "logs"
SESSIONS_FILE = STORAGE_DIR / "sessions.json"

# Image storage — one subdirectory per chapter
# Structure:  backend/images/ch01/img_p52_i0.jpg
IMAGES_DIR     = BASE_DIR / "images"
IMAGE_MANIFEST = IMAGES_DIR / "manifest.json"

STORAGE_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
IMAGES_DIR.mkdir(exist_ok=True)

# ── API KEYS ──────────────────────────────────────────────────────────────────
COHERE_API_KEY    = os.getenv("COHERE_API_KEY",    "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
PINECONE_API_KEY  = os.getenv("PINECONE_API_KEY",  "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# ── PINECONE ──────────────────────────────────────────────────────────────────
# One index, two namespaces — one per embedding model.
# Both models are 1024-dim so one index works for both.
# Separate namespaces prevent vector space mixing.
#
# Index creation requirements:
#   dimension = 1024, metric = cosine
#   integrated_model = "llama-text-embed-v2"  (enables Pinecone Inference)

PINECONE_INDEX_NAME  = os.getenv("PINECONE_INDEX_NAME", "gnosis-biology")
PINECONE_NS_COHERE   = os.getenv("PINECONE_NS_COHERE",  "campbell-biology-cohere")
PINECONE_NS_LLAMA    = os.getenv("PINECONE_NS_LLAMA",   "campbell-biology-llama")
PINECONE_DIMENSION   = 1024
PINECONE_METRIC      = "cosine"
PINECONE_EMBED_MODEL = "llama-text-embed-v2"  # must match your index config

# ── EMBEDDER ──────────────────────────────────────────────────────────────────
EMBEDDER_MODE = "llama_pinecone"   # <── CHANGE THIS    : llama_pinecone : cohere

EMBEDDING_CONFIG = {
    "cohere": {
        "provider"        : "cohere",
        "model"           : "embed-english-v3.0",
        "dimension"       : 1024,
        "namespace"       : PINECONE_NS_COHERE,
        "query_input_type": "search_query",
        "doc_input_type"  : "search_document",
    },
    "llama_pinecone": {
        "provider"  : "pinecone_inference",
        "model"     : PINECONE_EMBED_MODEL,
        "dimension" : 1024,
        "namespace" : PINECONE_NS_LLAMA,
        # Pinecone generates embeddings — no local model needed
        # Upsert: index.upsert_records(records=[{"_id":..., "text":..., ...}])
        # Query:  index.search(query={"inputs": {"text": "..."}, "top_k": N})
    },
    "huggingface": {
        "provider"  : "huggingface",
        "model"     : "all-MiniLM-L6-v2",
        "dimension" : 384,
        "namespace" : PINECONE_NS_COHERE,
    },
}
# ── LLM ───────────────────────────────────────────────────────────────────────
LLM_PROVIDER = "openrouter"   # ollama, cohere, ollama,  openrouter

LLM_CONFIG = {
    "anthropic": {
        "primary" : {"name": "claude-haiku-4-5-20251001", "temperature": 0.2},
        "summary" : {"name": "claude-haiku-4-5-20251001", "temperature": 0.3},
    },
    "cohere": {
        "primary" : {"name": "command-r-plus-08-2024", "temperature": 0.2},
        "summary" : {"name": "command-r-plus-08-2024",      "temperature": 0.3},
    },
    "ollama": {
        "primary" : {"name": "llama3.2:1b",  "temperature": 0.2},
        "summary" : {"name": "llama3.2:1b",  "temperature": 0.3},
        "base_url": "http://localhost:11434",
    },
    "openrouter": {
        "primary" : {"name": "openrouter/free", "temperature": 0.7},
        "summary" : {"name": "openrouter/free", "temperature": 0.3},
        "base_url": "https://openrouter.ai/api/v1",  
    },
}

# ── RETRIEVAL ─────────────────────────────────────────────────────────────────
RETRIEVAL_CONFIG = {
    "top_k_chunks"              : 3,
    "top_k_memory"              : 4,
    "cache_similarity_threshold": 1.0,
    # Image filtering — below these thresholds = decorative element, skip
    "min_image_pixels": 2500,   # width * height minimum
    "min_image_dim"   : 60,     # min px for either dimension
}

# ── SESSION STORE ─────────────────────────────────────────────────────────────
MAX_SESSIONS = 20

# ── BEHAVIOUR ─────────────────────────────────────────────────────────────────
VERBOSE_OUTPUT   = True
SOCRATIC_DEFAULT = True

# ── IMAGE SERVING ─────────────────────────────────────────────────────────────
# FastAPI mounts /images → backend/images/  (see api.py)
# Frontend fetches: http://localhost:8000/images/ch01/img_p52_i0.jpg
IMAGE_URL_PREFIX = "/images"

# ── SERVER ────────────────────────────────────────────────────────────────────
API_HOST = "127.0.0.1"
API_PORT = 8000

# ── LOGGING ───────────────────────────────────────────────────────────────────
LOG_LEVEL       = "INFO"
LOG_FILE        = LOGS_DIR / "gnosis.log"
LOG_FORMAT      = "[%(asctime)s] %(levelname)-8s - %(name)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# ── CHROMADB (Previously Missing) ───────────────────────────────────────────────
CHROMA_DB_PATH       = str(STORAGE_DIR / "chroma_db")
CHROMA_SEARCH_METRIC = "cosine"
COLLECTIONS = {
    "cache" : "gnosis_semantic_cache",
    "memory": "gnosis_long_term_memory",
}
