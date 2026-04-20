# context.py
# ─────────────────────────────────────────────────────────────────────────────
# Retrieval layer — gathers all context the LLM needs for a grounded answer.
#
# Routes to the correct search path based on EMBEDDER_MODE:
#   "cohere"/"huggingface" → embed locally first, then call search_chunks(vector=...)
#   "llama_pinecone"       → pass raw text, let Pinecone embed it internally
#
# Dependency direction:
#   context.py → models.py, pinecone_db.py, storage.py
# ─────────────────────────────────────────────────────────────────────────────

import logging
from config import EMBEDDER_MODE
from pinecone_db import search_chunks
import storage

logger = logging.getLogger(__name__)


# ── CACHE CHECK ───────────────────────────────────────────────────────────────

def check_cache(query: str) -> tuple[bool, str | None]:
    """
    Check ChromaDB semantic cache before running the full pipeline.
    Returns (True, answer) on hit, (False, None) on miss.
    """
    answer = storage.get_cached_answer(query)
    if answer:
        logger.info("Cache HIT — skipping Pinecone + LLM")
        return True, answer
    return False, None


# ── TEXTBOOK RETRIEVAL ────────────────────────────────────────────────────────

def get_textbook_context(query: str, namespace: str = None) -> dict:
    """
    Retrieve the top-k most relevant textbook chunks for this query.

    Automatically routes to the correct embedding path:
    - Cohere/HuggingFace: embeds locally, sends vector to Pinecone
    - Llama/Pinecone Inference: sends raw text, Pinecone embeds on server

    Returns {"text": str, "citations": list[dict]}
    where citations include image_files for the frontend to render figures.
    """
    logger.info(f"Retrieving textbook context [{EMBEDDER_MODE}]")

    if EMBEDDER_MODE == "llama_pinecone":
        # Pinecone Inference path — no local embedding step
        return search_chunks(query_text=query, namespace=namespace)
    else:
        # Standard path — embed locally, then send vector
        from models import embedder
        query_vec = embedder.embed_query(query)
        return search_chunks(query_vector=query_vec, namespace=namespace)


# ── MEMORY RETRIEVAL ──────────────────────────────────────────────────────────

def get_memory_context(query: str) -> str:
    """
    Retrieve relevant past-session summaries from ChromaDB.
    Always uses local embedding (ChromaDB doesn't support Pinecone Inference).
    """
    logger.info("Fetching memory context from ChromaDB")
    return storage.retrieve_memories(query)


# ── COMBINED FETCH ────────────────────────────────────────────────────────────

def fetch_all_context(query: str, namespace: str = None) -> dict:
    """
    Single call that returns everything main.py needs.

    Returns:
        {
            "query"            : str,         original question
            "document_context" : str,         textbook chunks (for LLM prompt)
            "citations"        : list[dict],  source metadata + image paths (for frontend)
            "past_knowledge"   : str,         session memory (for LLM prompt)
        }
    """
    logger.info(f"Fetching all context for: '{query[:80]}'")

    chunk_result = get_textbook_context(query, namespace=namespace)
    memory_text  = get_memory_context(query)

    return {
        "query"            : query,
        "document_context" : chunk_result["text"],
        "citations"        : chunk_result["citations"],
        "past_knowledge"   : memory_text,
    }
