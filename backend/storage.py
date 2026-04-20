# storage.py
# ─────────────────────────────────────────────────────────────────────────────
# All ChromaDB operations in one place.
# ChromaDB runs locally on disk — no external service needed.
#
# Two collections:
#   gnosis_semantic_cache    — avoids re-calling the LLM for near-identical questions
#   gnosis_long_term_memory  — stores compressed summaries of past study sessions
#
# Embedding for ChromaDB always uses the local embedder (Cohere or HuggingFace).
# When EMBEDDER_MODE = "llama_pinecone", we use the PineconeInferenceEmbedder
# which calls Pinecone's /embed endpoint to get vectors for ChromaDB storage.
# This way, ChromaDB memory is always searchable regardless of which mode is active.
# ─────────────────────────────────────────────────────────────────────────────

import logging
import chromadb

from config import (
    CHROMA_DB_PATH,
    CHROMA_SEARCH_METRIC,
    COLLECTIONS,
    RETRIEVAL_CONFIG,
)
from models import embedder   # works for all three embedder modes

logger = logging.getLogger(__name__)


# ── CHROMADB CLIENT ───────────────────────────────────────────────────────────
# PersistentClient writes to CHROMA_DB_PATH on disk. Data survives restarts.

_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)

_cache_col = _client.get_or_create_collection(
    name    = COLLECTIONS["cache"],
    metadata= {"hnsw:space": CHROMA_SEARCH_METRIC},
)
_memory_col = _client.get_or_create_collection(
    name    = COLLECTIONS["memory"],
    metadata= {"hnsw:space": CHROMA_SEARCH_METRIC},
)

logger.info(
    f"ChromaDB ready | "
    f"cache: {_cache_col.count()} items | "
    f"memory: {_memory_col.count()} items | "
    f"path: {CHROMA_DB_PATH}"
)


# ── SEMANTIC CACHE ────────────────────────────────────────────────────────────

def get_cached_answer(query: str) -> str | None:
    """
    Return a cached answer if a near-duplicate of this question was asked before.
    Cosine distance threshold is set in RETRIEVAL_CONFIG.cache_similarity_threshold.
    Returns None if this is a genuinely new question.
    """
    threshold = RETRIEVAL_CONFIG["cache_similarity_threshold"]

    # embed_query works for all embedder modes including llama_pinecone
    vec = embedder.embed_query(query)

    results = _cache_col.query(query_embeddings=[vec], n_results=1)
    if not results["distances"] or not results["distances"][0]:
        return None

    distance = results["distances"][0][0]
    if distance > threshold:
        logger.info(f"Cache HIT (distance: {distance:.4f} < threshold: {threshold})")
        return results["metadatas"][0][0].get("answer")

    logger.info(f"Cache MISS (distance: {distance:.4f})")
    return None


def cache_answer(query: str, answer: str) -> None:
    """Store a Q&A pair so future similar questions are served instantly."""
    vec = embedder.embed_query(query)
    _cache_col.upsert(
        ids       = [query[:512]],
        embeddings= [vec],
        metadatas = [{"answer": answer}],
    )
    logger.info("Answer cached in ChromaDB")


# ── LONG-TERM MEMORY ──────────────────────────────────────────────────────────

def retrieve_memories(query: str, top_k: int = None) -> str:
    """
    Find the most relevant past session summaries for the current question.
    Injects them into the LLM system prompt for personalisation.
    """
    top_k = top_k or RETRIEVAL_CONFIG["top_k_memory"]
    vec   = embedder.embed_query(query)

    results = _memory_col.query(query_embeddings=[vec], n_results=top_k)
    if results["documents"] and results["documents"][0]:
        memories = results["documents"][0]
        logger.info(f"Memory: retrieved {len(memories)} summaries")
        return "\n".join(memories)

    return "No prior study sessions recorded yet."


def store_memory(summary: str) -> None:
    """Persist a session summary. Hash of text is used as ID to avoid duplicates."""
    vec = embedder.embed_query(summary)
    _memory_col.upsert(
        ids       = [str(hash(summary))],
        embeddings= [vec],
        documents = [summary],
    )
    logger.info(f"Memory stored: '{summary[:80]}…'")


# ── DIAGNOSTICS ───────────────────────────────────────────────────────────────

def get_stats() -> dict:
    return {
        "cache_items" : _cache_col.count(),
        "memory_items": _memory_col.count(),
        "db_path"     : CHROMA_DB_PATH,
    }
