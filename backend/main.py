# main.py
# ─────────────────────────────────────────────────────────────────────────────
# Orchestrator: coordinates the full request lifecycle for every API call.
# This is the ONLY file that knows the correct sequence:
#   cache check → context retrieval → pipeline → storage → return
#
# api.py imports the `orchestrator` singleton from the bottom of this file.
# ─────────────────────────────────────────────────────────────────────────────

import logging
from datetime import datetime
from typing import Optional

import context
import pipeline
import storage
import sessions as session_store
from config import VERBOSE_OUTPUT

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)-8s - %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


class GnosisOrchestrator:
    """
    Coordinates a complete Gnosis RAG session.
    One instance is shared across the entire API (see bottom of file).

    Public methods:
        chat()            — process one student question, return answer + citations
        archive_session() — summarise a session and store it to long-term memory
        get_status()      — lightweight health check
    """

    # ── CHAT ──────────────────────────────────────────────────────────────────

    def chat(
        self,
        question    : str,
        chat_history: Optional[list] = None,
        session_id  : Optional[str]  = None,
        socratic    : bool = False,
        # namespace   : Optional[str]  = None,
    ) -> dict:
        """
        Process one student question end-to-end.

        Args:
            question     : The student's typed question.
            chat_history : Previous turns as list of {"role":..., "content":...}.
            session_id   : Frontend-generated UUID for the current session.
            socratic     : If True, the AI guides rather than directly answers.
            namespace    : Pinecone namespace (which book). Defaults to config default.

        Returns a dict:
            {
                "response"    : str,         — the answer text
                "citations"   : list[dict],  — source cards for the frontend
                "from_cache"  : bool,        — True if served from semantic cache
                "elapsed"     : float,       — seconds taken
                "error"       : bool,        — present and True if something went wrong
            }
        """
        logger.info("=" * 60)
        logger.info(f"QUERY: {question[:100]}")
        logger.info(f"SESSION: {session_id or 'none'} | SOCRATIC: {socratic}")
        logger.info("=" * 60)

        # Guard: empty or non-string input
        if not question or not isinstance(question, str):
            return {
                "response"  : "Please enter a valid question.",
                "citations" : [],
                "from_cache": False,
                "error"     : True,
            }

        chat_history = chat_history or []
        start        = datetime.now()

        try:
            # ── STEP 1: Semantic cache check ──────────────────────────────────
            # If we've answered a very similar question before, serve it instantly
            # without calling Pinecone or the LLM.
            is_cached, cached_answer = context.check_cache(question)
            if is_cached:
                elapsed = (datetime.now() - start).total_seconds()
                logger.info(f"Served from cache in {elapsed:.3f}s")
                return {
                    "response"   : cached_answer,
                    "citations"  : [],         # cache doesn't store citation metadata
                    "from_cache" : True,
                    "elapsed"    : round(elapsed, 2),
                }

            # ── STEP 2: Retrieve all context ──────────────────────────────────
            # Embeds the question, queries Pinecone for textbook chunks,
            # and retrieves relevant session memories from ChromaDB.
            ctx = context.fetch_all_context(question)

            if VERBOSE_OUTPUT:
                logger.info(f"Textbook context preview:\n{ctx['document_context'][:400]}")
                logger.info(f"Memory context:\n{ctx['past_knowledge'][:200]}")

            # ── STEP 3: Run RAG chain ─────────────────────────────────────────
            # Pass pre-fetched context into the LangChain LCEL pipeline.
            # pipeline.py selects the right prompt based on socratic flag.
            response = pipeline.run_rag(
                question        =question,
                chat_history    =chat_history,
                document_context=ctx["document_context"],
                past_knowledge  =ctx["past_knowledge"],
                socratic_mode   =socratic,
            )

            # ── STEP 4: Cache the new answer ──────────────────────────────────
            # Store so near-duplicate questions can be served without re-calling LLM.
            storage.cache_answer(question, response)

            # ── STEP 5: Register session ──────────────────────────────────────
            # Create or update the session record in sessions.json for the sidebar.
            if session_id:
                session_store.upsert_session(
                    session_id=session_id,
                    summary=question[:100],   # first question becomes the session title
                )

            elapsed = (datetime.now() - start).total_seconds()
            logger.info(f"Response ready in {elapsed:.3f}s — {len(ctx['citations'])} citations")

            return {
                "response"    : response,
                "citations"   : ctx["citations"],
                "from_cache"  : False,
                "elapsed"     : round(elapsed, 2),
                "had_context" : bool(ctx["document_context"]),
                "had_memory"  : "No prior" not in ctx["past_knowledge"],
            }

        except Exception as e:
            logger.error(f"Error in chat(): {e}", exc_info=True)
            return {
                "response"  : f"Something went wrong: {str(e)}. Please try again.",
                "citations" : [],
                "from_cache": False,
                "error"     : True,
            }

    # ── SESSION ARCHIVING ─────────────────────────────────────────────────────

    def archive_session(self, chat_history: list, session_id: Optional[str] = None) -> dict:
        """
        Archive a completed session:
          1. Compress the chat transcript into bullet-point memory using the LLM.
          2. Store the summary in ChromaDB long-term memory for future sessions.
          3. Update the session record in sessions.json with the LLM summary.

        Args:
            chat_history : Full list of {"role":..., "content":...} turns.
            session_id   : UUID of the session to mark as archived.

        Returns:
            {"success": bool, "summary": str | None}
        """
        logger.info("=" * 60)
        logger.info("ARCHIVING SESSION")
        logger.info("=" * 60)

        if not chat_history:
            logger.info("No history to archive")
            return {"success": True, "summary": None}

        # Format chat history into a readable transcript for the LLM
        lines = []
        for msg in chat_history:
            role    = msg.get("role", "unknown").upper()
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        transcript = "\n".join(lines)

        # Generate compressed summary
        summary = pipeline.summarise_transcript(transcript)
        if not summary:
            logger.error("Summarisation returned nothing — archive aborted")
            return {"success": False, "summary": None}

        # Persist to ChromaDB long-term memory
        try:
            storage.store_memory(summary)
            logger.info("Session persisted to ChromaDB memory")
        except Exception as e:
            logger.error(f"Memory storage failed: {e}", exc_info=True)
            return {"success": False, "summary": summary}

        # Update session record with the LLM summary
        if session_id:
            session_store.mark_archived(session_id, summary)

        logger.info("Session archived successfully")
        return {"success": True, "summary": summary}

    # ── HEALTH CHECK ──────────────────────────────────────────────────────────

    def get_status(self) -> dict:
        """Lightweight health check — returns storage stats and system state."""
        from pinecone_db import get_index_stats
        return {
            "status"    : "healthy",
            "timestamp" : datetime.now().isoformat(),
            "chromadb"  : storage.get_stats(),
            "pinecone"  : get_index_stats(),
        }


# ── GLOBAL SINGLETON ─────────────────────────────────────────────────────────
# api.py does: from main import orchestrator

orchestrator = GnosisOrchestrator()
logger.info(f"Gnosis orchestrator ready")
