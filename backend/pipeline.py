# pipeline.py
# ─────────────────────────────────────────────────────────────────────────────
# LCEL chain definitions and execution helpers.
# Chains receive pre-fetched context — NO retrieval happens here.
# Retrieval is done in context.py; chains just format inputs → LLM → parse output.
#
# Why separate chains from retrieval?
# It makes each unit testable in isolation:
#   - Test context.py by checking what chunks Pinecone returns.
#   - Test pipeline.py by injecting fake context and checking the LLM output.
#
# Dependency direction (imports only downward):
#   pipeline.py → models.py, prompts.py
# ─────────────────────────────────────────────────────────────────────────────

import logging
from langchain_core.runnables import RunnablePassthrough,RunnableLambda
from langchain_core.output_parsers import StrOutputParser

from models import primary_llm, summary_llm
from prompts import GNOSIS_PROMPT, SOCRATIC_PROMPT, SUMMARY_PROMPT

logger = logging.getLogger(__name__)


# ── RAG CHAIN — NORMAL MODE ───────────────────────────────────────────────────
# Input keys: question, chat_history, document_context, past_knowledge
# RunnablePassthrough() passes the input dict straight to the prompt template.
# The prompt fills its variables, the LLM generates a response, StrOutputParser
# strips the LangChain message wrapper and returns a plain string.


# ===============
# NEW-FUNCTION-TO-LOG-LLM-MODEL
def log_model_routing(output):
    """Peeks at the LLM message to log which specific model OpenRouter used."""
    if hasattr(output, 'response_metadata'):
        # Check 'model_name' first, then fallback to 'model'
        meta = output.response_metadata
        actual_model = meta.get("model_name") or meta.get("model") or "openrouter/free"
        
        logger.info(f"OpenRouter routed to: {actual_model}")
        print(f"\n[OpenRouter Routing] Answered by: {actual_model}\n")
        
    return output
# =============



_rag_chain = (
    RunnablePassthrough()
    | GNOSIS_PROMPT
    | primary_llm
    | RunnableLambda(log_model_routing)
    | StrOutputParser()
)

# _rag_chain = (
#     RunnablePassthrough()

#     | GNOSIS_PROMPT
#     | primary_llm
#     | RunnableLambda(log_model_routing) # <── Wired in here
#     | StrOutputParser()
# )


# ── RAG CHAIN — SOCRATIC MODE ─────────────────────────────────────────────────
# Same structure, different prompt — guides instead of directly answering.

_socratic_chain = (
    RunnablePassthrough()
    | SOCRATIC_PROMPT
    | primary_llm
    | RunnableLambda(log_model_routing)
    | StrOutputParser()
)


# ── SUMMARY CHAIN ─────────────────────────────────────────────────────────────
# Input key: transcript (full session text)
# Uses summary_llm (can be a cheaper/smaller model than primary_llm)

_summary_chain = (
    SUMMARY_PROMPT
    | summary_llm
    | StrOutputParser()
)


# ── RUNNER FUNCTIONS ──────────────────────────────────────────────────────────
# main.py calls these. It does not touch chain objects directly.

def run_rag(
    question        : str,
    chat_history    : list,
    document_context: str,
    past_knowledge  : str,
    socratic_mode   : bool = False,
) -> str:
    """
    Run the main study RAG chain.

    Args:
        question         : The student's question.
        chat_history     : List of {"role": ..., "content": ...} dicts.
        document_context : Retrieved textbook chunk text from Pinecone.
        past_knowledge   : Session memory summaries from ChromaDB.
        socratic_mode    : If True, uses the Socratic prompt (guides, doesn't answer directly).

    Returns:
        The LLM's response as a plain string.

    Raises:
        Exception — propagated to main.py which catches and wraps it.
    """
    # Pick which chain to use based on mode
    chain = _socratic_chain if socratic_mode else _rag_chain
    mode  = "SOCRATIC" if socratic_mode else "DIRECT"

    logger.info(f"Invoking RAG chain [{mode} mode]")

    return chain.invoke({
        "question"        : question,
        "chat_history"    : chat_history,
        "document_context": document_context,
        "past_knowledge"  : past_knowledge,
    })


def summarise_transcript(transcript: str) -> str | None:
    """
    Compress a full chat transcript into 3–5 memory bullets.
    Called during session archiving (POST /api/archive).

    Returns:
        Bullet-point summary string, or None if transcript is empty or fails.
    """
    if not transcript.strip():
        logger.warning("Empty transcript — summarisation skipped")
        return None

    logger.info("Invoking summary chain")
    try:
        return _summary_chain.invoke({"transcript": transcript})
    except Exception as e:
        logger.error(f"Summarisation failed: {e}", exc_info=True)
        return None
