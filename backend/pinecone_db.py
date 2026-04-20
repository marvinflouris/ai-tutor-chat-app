# pinecone_db.py
# ─────────────────────────────────────────────────────────────────────────────
# All Pinecone operations: connecting, searching, upserting, and listing docs.
#
# ── NAMESPACE SAFETY ──────────────────────────────────────────────────────────
# The active namespace is resolved ONCE at module import from EMBEDDER_MODE:
#
#   EMBEDDER_MODE = "cohere"         → _ACTIVE_NS = PINECONE_NS_COHERE
#   EMBEDDER_MODE = "llama_pinecone" → _ACTIVE_NS = PINECONE_NS_LLAMA
#   EMBEDDER_MODE = "huggingface"    → _ACTIVE_NS = PINECONE_NS_COHERE
#
# This guarantees that the same embedding model used to index the vectors is
# also used when querying — preventing the "wrong namespace" mismatch bug where
# Cohere vectors are searched with a Llama query vector (or vice versa).
#
# ── TWO SEARCH PATHS ─────────────────────────────────────────────────────────
#   "cohere" / "huggingface"  →  _search_standard()
#     Your code generates the embedding → sends vector to Pinecone
#     index.query(vector=[...], namespace=_ACTIVE_NS, include_metadata=True)
#
#   "llama_pinecone"          →  _search_pinecone_inference()
#     Raw text is sent → Pinecone generates the embedding via Llama Text Embed v2
#     index.search(namespace=_ACTIVE_NS, query={"inputs":{"text":...}, "top_k":N})
#     REQUIRES: index created with integrated_model = "llama-text-embed-v2"
#
# Both paths return identical output: {"text": str, "citations": list[dict]}
# ─────────────────────────────────────────────────────────────────────────────

import logging
from typing import List, Optional

from pinecone import Pinecone

from config import (
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    PINECONE_NS_COHERE,
    PINECONE_NS_LLAMA,
    PINECONE_EMBED_MODEL,
    EMBEDDER_MODE,
    RETRIEVAL_CONFIG,
    IMAGE_URL_PREFIX,
)

logger = logging.getLogger(__name__)


# ── CLIENT ────────────────────────────────────────────────────────────────────

if not PINECONE_API_KEY:
    raise EnvironmentError(
        "PINECONE_API_KEY is not set. Add it to your .env file.\n"
        "Get your key from: https://app.pinecone.io"
    )

_pc    = Pinecone(api_key=PINECONE_API_KEY)
_index = _pc.Index(PINECONE_INDEX_NAME)

# ── NAMESPACE AUTO-RESOLUTION ─────────────────────────────────────────────────
# Resolved at import time — guaranteed to match the embedder actually in use.
# Never override this with a hardcoded string; always go through this variable.
_ACTIVE_NS = PINECONE_NS_LLAMA if EMBEDDER_MODE == "llama_pinecone" else PINECONE_NS_COHERE

logger.info(
    f"Pinecone ready | index='{PINECONE_INDEX_NAME}' | "
    f"embedder='{EMBEDDER_MODE}' | namespace='{_ACTIVE_NS}'"
)


# ── DOCUMENT REGISTRY ─────────────────────────────────────────────────────────
# Add an entry here each time you embed a new book.
# Exposed via GET /api/documents → drives the + document picker in the frontend.

AVAILABLE_DOCUMENTS = [
    {
        "id"              : "campbell-biology-12e",
        "title"           : "Campbell Biology",
        "subtitle"        : "12th Edition",
        "description"     : "Urry, Cain, Wasserman, Minorsky, Orr — Units 1–8, Chapters 1–56",
        "namespace_cohere": PINECONE_NS_COHERE,
        "namespace_llama" : PINECONE_NS_LLAMA,
        "color"           : "#3d6b3d",
    },
    # ── ADD NEW BOOKS HERE ───────────────────────────────────────────────────
    # {
    #     "id"              : "lehninger-biochemistry-7e",
    #     "title"           : "Lehninger Principles of Biochemistry",
    #     "subtitle"        : "7th Edition",
    #     "description"     : "Nelson & Cox — Chapters 1–28",
    #     "namespace_cohere": "lehninger-cohere",
    #     "namespace_llama" : "lehninger-llama",
    #     "color"           : "#3d4b6b",
    # },
]


# ── CITATION BUILDER ──────────────────────────────────────────────────────────

def _build_citation(i: int, meta: dict, score: float) -> dict:
    """
    Convert Pinecone match metadata into a structured citation dict for the frontend.

    image_files paths are normalised to URL form:
        "ch07/img_p176_i0.jpg"  →  "/images/ch07/img_p176_i0.jpg"
    """
    raw_imgs = meta.get("image_files", [])
    # Deserialise if stored as a JSON string (Pinecone Inference path)
    if isinstance(raw_imgs, str):
        import json
        try: raw_imgs = json.loads(raw_imgs)
        except: raw_imgs = []

    image_urls = []
    for f in raw_imgs:
        if not isinstance(f, str): continue
        clean = f.replace("\\", "/").lstrip("/")
        if not clean.startswith("images/"):
            clean = f"images/{clean}"
        image_urls.append(f"{IMAGE_URL_PREFIX}/{clean.removeprefix('images/')}")

    # figure_refs may also be a JSON string
    fig_refs = meta.get("figure_refs", [])
    if isinstance(fig_refs, str):
        import json
        try: fig_refs = json.loads(fig_refs)
        except: fig_refs = []

    return {
        "ref"         : i,
        "title"       : meta.get("title",        "Campbell Biology"),
        "unit"        : meta.get("unit",          ""),
        "chapter"     : meta.get("chapter",       ""),
        "chapter_title": meta.get("chapter_title",""),
        "concept"     : meta.get("concept",       ""),
        "concept_title": meta.get("concept_title",""),
        "section"     : meta.get("section",       ""),
        "page"        : meta.get("page",          ""),
        "doc_id"      : meta.get("doc_id",        "campbell-biology-12e"),
        "score"       : round(score, 3),
        "figure_refs" : fig_refs,
        "image_files" : image_urls,
    }


# ── STANDARD SEARCH (Cohere / HuggingFace) ────────────────────────────────────

def _search_standard(query_vector: List[float], top_k: int, namespace: str) -> dict:
    """Vector search using a locally-generated embedding (Cohere or HuggingFace)."""
    results = _index.query(
        vector          = query_vector,
        top_k           = top_k,
        include_metadata= True,
        namespace       = namespace,
    )
    matches = results.get("matches", [])
    if not matches:
        logger.warning(f"No matches returned from Pinecone (ns={namespace})")
        return {"text": "No relevant content found in the textbook.", "citations": []}

    lines, citations = [], []
    for i, m in enumerate(matches, 1):
        meta  = m.get("metadata", {})
        chunk = meta.get("text", "")
        score = m.get("score", 0.0)
        ref   = f"[{i}] {meta.get('title','Campbell Biology')}"
        for f in ("chapter","concept","section"):
            v = meta.get(f,"")
            if v: ref += f" · {v}"
        p = meta.get("page","")
        if p: ref += f" · p.{p}"
        lines.append(f"{ref}\n{chunk}")
        citations.append(_build_citation(i, meta, score))

    return {"text": "\n\n---\n\n".join(lines), "citations": citations}


# ── PINECONE INFERENCE SEARCH (Llama Text Embed v2) ───────────────────────────

def _search_pinecone_inference(query_text: str, top_k: int, namespace: str) -> dict:
    """
    Vector search using Pinecone's integrated Llama Text Embed v2.
    Pinecone embeds the query text on its servers — no local embedding needed.
    Requires the index to have been created with an integrated embedding model.
    """
    try:
        results = _index.search(
            namespace = namespace,
            query     = {"inputs": {"text": query_text}, "top_k": top_k},
            fields    = ["text","title","unit","chapter","chapter_title","concept",
                         "concept_title","section","page","doc_id","figure_refs","image_files"],
        )
    except AttributeError:
        logger.error(
            "index.search() not available. Upgrade: pip install --upgrade pinecone-client  "
            "(requires pinecone-client >= 4.0)"
        )
        return {"text": "Pinecone Inference search unavailable — upgrade pinecone-client to v4+.", "citations": []}

    hits = results.get("result", {}).get("hits", [])
    if not hits:
        logger.warning(f"No hits from Pinecone Inference (ns={namespace})")
        return {"text": "No relevant content found in the textbook.", "citations": []}

    lines, citations = [], []
    for i, hit in enumerate(hits, 1):
        fields = hit.get("fields", {})
        score  = hit.get("_score", 0.0)
        chunk  = fields.get("text", "")
        ref    = f"[{i}] {fields.get('title','Campbell Biology')}"
        for f in ("chapter","concept","section"):
            v = fields.get(f,"")
            if v: ref += f" · {v}"
        p = fields.get("page","")
        if p: ref += f" · p.{p}"
        lines.append(f"{ref}\n{chunk}")
        citations.append(_build_citation(i, fields, score))

    return {"text": "\n\n---\n\n".join(lines), "citations": citations}


# ── PUBLIC SEARCH INTERFACE ───────────────────────────────────────────────────

def search_chunks(
    query_vector: Optional[List[float]] = None,
    query_text  : Optional[str]  = None,
    top_k       : int  = None,
    namespace   : str  = None,
) -> dict:
    """
    Route to the correct search path based on EMBEDDER_MODE.

    Args:
        query_vector : Pre-generated embedding vector (Cohere / HuggingFace path).
        query_text   : Raw question text (Llama / Pinecone Inference path).
        top_k        : Chunks to retrieve. Defaults to RETRIEVAL_CONFIG value.
        namespace    : Pinecone namespace. Defaults to _ACTIVE_NS (auto-resolved).
                       ⚠ Do NOT pass a hardcoded namespace — let the default ensure
                         the namespace always matches the active embedder.

    Returns:
        {"text": str, "citations": list[dict]}
    """
    top_k     = top_k     or RETRIEVAL_CONFIG["top_k_chunks"]
    namespace = namespace or _ACTIVE_NS    # ← always matches EMBEDDER_MODE

    try:
        if EMBEDDER_MODE == "llama_pinecone":
            if not query_text:
                raise ValueError("query_text required for llama_pinecone mode")
            logger.info(f"Pinecone Inference search | ns={namespace} | top_k={top_k}")
            result = _search_pinecone_inference(query_text, top_k, namespace)
        else:
            if query_vector is None:
                raise ValueError("query_vector required for cohere/huggingface mode")
            logger.info(f"Standard vector search | ns={namespace} | top_k={top_k}")
            result = _search_standard(query_vector, top_k, namespace)

        n_imgs = sum(len(c.get("image_files",[])) for c in result["citations"])
        logger.info(f"Retrieved {len(result['citations'])} chunks | {n_imgs} image references")
        return result

    except Exception as e:
        logger.error(f"Pinecone search failed: {e}", exc_info=True)
        return {"text": "Textbook database temporarily unavailable.", "citations": []}


# ── UPSERT HELPERS (embed_campbell.py) ────────────────────────────────────────

def upsert_cohere_vectors(vectors: list, namespace: str = None) -> None:
    """Upsert pre-computed Cohere embedding vectors into Pinecone."""
    ns = namespace or PINECONE_NS_COHERE
    _index.upsert(vectors=vectors, namespace=ns)
    logger.info(f"Upserted {len(vectors)} Cohere vectors → ns='{ns}'")


def upsert_llama_records(records: list, namespace: str = None) -> None:
    """
    Upsert text records for Pinecone Inference embedding (Llama Text Embed v2).
    Pinecone reads the 'text' field and generates the embedding automatically.
    """
    ns = namespace or PINECONE_NS_LLAMA
    _index.upsert_records(records=records, namespace=ns)
    logger.info(f"Upserted {len(records)} Llama records → ns='{ns}'")


# ── INDEX STATS ───────────────────────────────────────────────────────────────

def get_index_stats() -> dict:
    """Return Pinecone index statistics for the /api/health endpoint."""
    try:
        stats = _index.describe_index_stats()
        if hasattr(stats, "total_vector_count"):
            total  = stats.total_vector_count
            ns_map = {k: v.vector_count for k, v in (stats.namespaces or {}).items()}
        else:
            total  = stats.get("total_vector_count", 0)
            ns_map = {k: v.get("vector_count", 0) for k, v in stats.get("namespaces", {}).items()}
        return {
            "total_vectors"  : total,
            "namespaces"     : ns_map,
            "active_namespace": _ACTIVE_NS,
            "active_embedder" : EMBEDDER_MODE,
        }
    except Exception as e:
        logger.error(f"Could not fetch Pinecone stats: {e}")
        return {"total_vectors": 0, "namespaces": {}, "active_namespace": _ACTIVE_NS}
