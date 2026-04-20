# models.py
# ─────────────────────────────────────────────────────────────────────────────
# Constructs and exposes AI model singletons used across the app.
#
# embedder     — converts text → 1024-dim vectors
# primary_llm  — answers student questions
# summary_llm  — compresses session transcripts
#
# Three embedder paths:
#   "cohere"          → calls Cohere API, returns vector locally
#   "llama_pinecone"  → Pinecone generates the vector natively (no local embedding)
#                       This returns a special proxy object — pinecone_db.py calls
#                       the Pinecone Inference API directly for both upsert and query.
#   "huggingface"     → runs all-MiniLM-L6-v2 locally, no API key needed (dev only)
# ─────────────────────────────────────────────────────────────────────────────

import logging
from config import (
    EMBEDDER_MODE, EMBEDDING_CONFIG,
    COHERE_API_KEY, ANTHROPIC_API_KEY, PINECONE_API_KEY,OPENROUTER_API_KEY,
    PINECONE_INDEX_NAME, PINECONE_EMBED_MODEL,
    LLM_PROVIDER, LLM_CONFIG,
)

logger = logging.getLogger(__name__)


# ── PINECONE INFERENCE EMBEDDER PROXY ─────────────────────────────────────────
# When EMBEDDER_MODE = "llama_pinecone", we do NOT generate embeddings locally.
# Pinecone Inference generates them on the server side.
#
# This proxy exposes the same .embed_query() interface as LangChain embedders
# so the rest of the codebase can call it identically regardless of mode.
# Internally, pinecone_db.py bypasses this for Pinecone vector operations
# and uses the native Pinecone Inference search API instead.

class PineconeInferenceEmbedder:
    """
    Proxy embedder for Pinecone's integrated inference (Llama Text Embed v2).

    For Pinecone operations (search, upsert), pinecone_db.py uses the
    Pinecone Inference API directly — this class only needs to handle
    ChromaDB operations (semantic cache + memory), where we need local vectors.

    We achieve this by calling Pinecone's embed endpoint to get the
    vector, which we then store in ChromaDB.
    """

    def __init__(self, api_key: str, model: str, index_name: str):
        from pinecone import Pinecone
        self._pc       = Pinecone(api_key=api_key)
        self._model    = model
        self._index    = self._pc.Index(index_name)
        logger.info(f"Embedder: Pinecone Inference / {model} (1024-dim)")

    def embed_query(self, text: str) -> list[float]:
        """
        Generate a 1024-dim embedding for a query string using Pinecone Inference.
        Used by ChromaDB cache and memory lookups.
        """
        # Pinecone Inference embed endpoint
        # input_type "query" vs "passage" adjusts the embedding for retrieval
        result = self._pc.inference.embed(
            model      = self._model,
            inputs     = [text],
            parameters = {"input_type": "query", "truncate": "END"},
        )
        return result[0].values   # list[float], length 1024

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """
        Batch-embed document chunks via Pinecone Inference.
        Used by embed_campbell.py for the Llama path.
        """
        result = self._pc.inference.embed(
            model      = self._model,
            inputs     = texts,
            parameters = {"input_type": "passage", "truncate": "END"},
        )
        return [item.values for item in result]


# ── EMBEDDER FACTORY ──────────────────────────────────────────────────────────

def _build_embedder():
    cfg = EMBEDDING_CONFIG[EMBEDDER_MODE]

    if cfg["provider"] == "cohere":
        # Cohere embed-english-v3.0 — 1024-dim
        # LangChain's CohereEmbeddings automatically passes input_type per method
        from langchain_cohere import CohereEmbeddings
        logger.info(f"Embedder: Cohere / {cfg['model']} ({cfg['dimension']}-dim)")
        return CohereEmbeddings(
            model=cfg["model"],
            cohere_api_key=COHERE_API_KEY,
        )

    elif cfg["provider"] == "pinecone_inference":
        # Nvidia Llama Text Embed v2 via Pinecone Inference API
        # Pinecone generates embeddings on its servers — no local model download
        return PineconeInferenceEmbedder(
            api_key    = PINECONE_API_KEY,
            model      = cfg["model"],
            index_name = PINECONE_INDEX_NAME,
        )

    elif cfg["provider"] == "huggingface":
        # all-MiniLM-L6-v2 — 384-dim, runs locally, no internet needed
        from langchain_huggingface import HuggingFaceEmbeddings
        logger.info(f"Embedder: HuggingFace / {cfg['model']} ({cfg['dimension']}-dim)")
        return HuggingFaceEmbeddings(
            model_name   = cfg["model"],
            model_kwargs = {"device": "cpu"},
            encode_kwargs= {"normalize_embeddings": True},
        )

    else:
        raise ValueError(f"Unknown embedder provider: {cfg['provider']}")


# ── LLM FACTORY ───────────────────────────────────────────────────────────────

def _build_llm(role: str):
    """role: 'primary' (answering) or 'summary' (session archiving)."""
    cfg       = LLM_CONFIG[LLM_PROVIDER]
    model_cfg = cfg[role]

    if LLM_PROVIDER == "anthropic":
        from langchain_anthropic import ChatAnthropic
        logger.info(f"LLM [{role}]: Anthropic / {model_cfg['name']}")
        return ChatAnthropic(
            model      = model_cfg["name"],
            temperature= model_cfg["temperature"],
            api_key    = ANTHROPIC_API_KEY,
        )

    elif LLM_PROVIDER == "cohere":
        from langchain_cohere import ChatCohere
        logger.info(f"LLM [{role}]: Cohere / {model_cfg['name']}")
        return ChatCohere(
            model         = model_cfg["name"],
            temperature   = model_cfg["temperature"],
            cohere_api_key= COHERE_API_KEY,
        )

    elif LLM_PROVIDER == "ollama":
        from langchain_community.llms import Ollama
        logger.info(f"LLM [{role}]: Ollama / {model_cfg['name']}")
        return Ollama(
            model      = model_cfg["name"],
            base_url   = cfg["base_url"],
            temperature= model_cfg["temperature"],
        )
    
    elif LLM_PROVIDER == "openrouter":
        from langchain_openai import ChatOpenAI
        logger.info(f"LLM [{role}]: OpenRouter / {model_cfg['name']}")
        return ChatOpenAI(
            model_name      = model_cfg["name"],
            temperature     = model_cfg["temperature"],
            openai_api_key  = OPENROUTER_API_KEY, # Matches your style
            openai_api_base = cfg["base_url"],
        )    

    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")


# ── SINGLETONS ────────────────────────────────────────────────────────────────
# Built once at import time, shared across all modules via Python module cache.

embedder    = _build_embedder()
primary_llm = _build_llm("primary")
summary_llm = _build_llm("summary")

logger.info("All models initialised successfully")
