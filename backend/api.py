# api.py
# ─────────────────────────────────────────────────────────────────────────────
# FastAPI server — HTTP bridge between the frontend and the RAG backend.
#
# Endpoints:
#   POST   /api/chat            — send a question, receive answer + citations + images
#   GET    /api/sessions        — recent study sessions for the sidebar
#   DELETE /api/sessions/:id    — delete a session
#   POST   /api/archive         — summarise + archive a completed session
#   GET    /api/documents       — list available books in Pinecone
#   GET    /api/health          — system health check
#
# Static:
#   GET    /images/**           — serve extracted figure images
#                                 e.g. /images/ch01/img_p52_i0.jpg
# ─────────────────────────────────────────────────────────────────────────────

import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from main import orchestrator
from pinecone_db import AVAILABLE_DOCUMENTS
import sessions as session_store
from config import API_HOST, API_PORT, IMAGES_DIR

logger = logging.getLogger(__name__)


# ── APP SETUP ─────────────────────────────────────────────────────────────────

app = FastAPI(
    title      = "Gnosis by Zoe — Study AI API",
    description= "RAG-powered biology study assistant backed by Campbell Biology + Pinecone",
    version    = "1.1.0",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins    = ["*"],   # lock down in production
    allow_credentials= True,
    allow_methods    = ["*"],
    allow_headers    = ["*"],
)

# ── STATIC IMAGE FILES ────────────────────────────────────────────────────────
# Mount the images directory so extracted figures are accessible at /images/**
# The frontend uses citation.image_files to construct these URLs.
# e.g.  /images/ch01/img_p52_i0.jpg
IMAGES_DIR.mkdir(exist_ok=True)
app.mount("/images", StaticFiles(directory=str(IMAGES_DIR)), name="images")


# ── REQUEST / RESPONSE MODELS ─────────────────────────────────────────────────

class ChatMessage(BaseModel):
    role   : str    # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    question   : str
    session_id : Optional[str]     = None
    history    : list[ChatMessage] = []
    socratic   : bool              = False
    # namespace  : Optional[str]     = None   # Pinecone namespace (book)

class ChatResponse(BaseModel):
    response   : str
    citations  : list[dict]        = []
    from_cache : bool              = False
    elapsed    : float             = 0.0
    session_id : Optional[str]     = None

class ArchiveRequest(BaseModel):
    session_id : str
    history    : list[ChatMessage]


# ── ENDPOINTS ─────────────────────────────────────────────────────────────────

@app.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Main chat endpoint.
    Returns answer text + citation metadata including image_files for figure display.
    """
    history_dicts = [{"role": m.role, "content": m.content} for m in req.history]
    result = orchestrator.chat(
        question    = req.question,
        chat_history= history_dicts,
        session_id  = req.session_id,
        socratic    = req.socratic,
        # namespace   = req.namespace,
    )
    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["response"])

    return ChatResponse(
        response  = result["response"],
        citations = result.get("citations",  []),
        from_cache= result.get("from_cache", False),
        elapsed   = result.get("elapsed",    0.0),
        session_id= req.session_id,
    )


@app.get("/api/sessions")
async def get_sessions():
    """Recent study sessions for the sidebar, newest first."""
    return {"sessions": session_store.get_all_sessions()}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str):
    found = session_store.delete_session(session_id)
    if not found:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"deleted": True, "session_id": session_id}


@app.post("/api/archive")
async def archive_session(req: ArchiveRequest):
    """
    Summarise and archive a completed session to long-term ChromaDB memory.
    """
    history_dicts = [{"role": m.role, "content": m.content} for m in req.history]
    result = orchestrator.archive_session(chat_history=history_dicts, session_id=req.session_id)
    if not result["success"]:
        raise HTTPException(status_code=500, detail="Session archiving failed")
    return {"success": True, "session_id": req.session_id, "summary": result.get("summary")}


@app.get("/api/documents")
async def get_documents():
    """
    List all available books in the Pinecone index.
    The frontend renders these in the document picker (+ button).
    """
    return {"documents": AVAILABLE_DOCUMENTS}


@app.get("/api/health")
async def health():
    """
    Health check — returns ChromaDB stats, Pinecone stats, and image count.
    """
    status = orchestrator.get_status()
    # Also report how many images have been saved
    img_count = sum(1 for _ in IMAGES_DIR.rglob("*.jpg"))
    status["images"] = {"saved_files": img_count, "images_dir": str(IMAGES_DIR)}
    return status


# ── DEV SERVER ENTRY POINT ────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn, logging as _log
    _log.basicConfig(level=_log.INFO, format="[%(asctime)s] %(levelname)-8s - %(name)s - %(message)s")

    print(f"\n{'='*60}")
    print(f"  Orelia by Zoe — Study AI API  v1.1")
    print(f"  Running at:    http://{API_HOST}:{API_PORT}")
    print(f"  API docs:      http://{API_HOST}:{API_PORT}/docs")
    print(f"  Image server:  http://{API_HOST}:{API_PORT}/images/ch01/...")
    print(f"{'='*60}\n")

    uvicorn.run("api:app", host=API_HOST, port=API_PORT, reload=True)
