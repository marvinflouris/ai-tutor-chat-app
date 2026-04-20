> [!NOTE]: THE PROJECT NAME WAS SHIFTED FROM "Gnosis by Zoe" to "Orelia by Enthios" because of some terms and conditions encountered therefore keep in mind where you see either of the namings interchanged it's because of that edit and shall be corrected in later versions.
> The html files have undergone a series of changes but are well configured to the same backend logic.
> Some frontend descriptions here may not be fully reflected in the html files due to constant editing that occured.  
# Gnosis by Zoe — File Roles & System Flow

## Every File and What It Does

```
gnosis_by_zoe/
│
├── GUIDE.md                ← This file. Roles + flow overview.
├── WORKFLOW.md             ← Deep-dive: setup, API reference, troubleshooting.
│
├── frontend/
│   └── Gnosis.html         ← The complete student-facing web app.
│                              Open this in Chrome. No server needed for the UI —
│                              it talks to the backend via fetch() calls.
│
└── backend/
    ├── .env                ← YOUR SECRET KEYS. Never commit. Copy from .env.template.
    ├── .env.template       ← Safe template — shows required key names, no values.
    ├── requirements.txt    ← Python packages. Run: pip install -r requirements.txt
    │
    ├── config.py           ── THE CONTROL PANEL. Every tunable setting lives here.
    │                          Change EMBEDDER_MODE, LLM_PROVIDER, top_k, image
    │                          filters, port number — all in one place.
    │                          No other file needs editing to switch providers.
    │
    ├── models.py           ── BUILDS THE AI OBJECTS.
    │                          Called once at startup. Creates:
    │                            embedder    — text → 1024-dim vector
    │                            primary_llm — answers student questions
    │                            summary_llm — compresses session memory
    │                          Reads config.py to know which provider to use.
    │
    ├── prompts.py          ── ALL INSTRUCTION TEXT FOR THE LLM.
    │                          GNOSIS_PROMPT   = direct answer mode
    │                          SOCRATIC_PROMPT = guide-not-answer mode
    │                          SUMMARY_PROMPT  = session compression
    │                          Edit here to change how Gnosis teaches.
    │
    ├── pinecone_db.py      ── ALL PINECONE OPERATIONS.
    │                          Connects, searches, builds citations.
    │                          Two search paths:
    │                            Cohere mode   → index.query(vector=[...])
    │                            Llama mode    → index.search(inputs.text=...)
    │                          AVAILABLE_DOCUMENTS registry = the + picker in UI.
    │
    ├── storage.py          ── ALL CHROMADB OPERATIONS (local, no API needed).
    │                          Collection 1: semantic_cache
    │                            Stores Q+A pairs. Near-identical questions
    │                            return instantly without calling Pinecone or LLM.
    │                          Collection 2: long_term_memory
    │                            Stores bullet summaries of past sessions.
    │                            Lets Gnosis remember what the student studied before.
    │
    ├── context.py          ── RETRIEVAL COORDINATOR.
    │                          Given a question:
    │                            1. Check cache (storage.py)
    │                            2. Embed question (models.py)
    │                            3. Search Pinecone (pinecone_db.py)
    │                            4. Search memory (storage.py)
    │                            5. Return everything bundled for the LLM
    │
    ├── pipeline.py         ── LANGCHAIN CHAINS.
    │                          Takes pre-fetched context → feeds prompt + LLM → returns string.
    │                          run_rag()              — answers with GNOSIS or SOCRATIC prompt
    │                          summarise_transcript() — condenses session for memory
    │
    ├── main.py             ── THE ORCHESTRATOR.
    │                          Knows the correct sequence for every request.
    │                          api.py calls this. Nothing else does.
    │                          chat()           — full request lifecycle
    │                          archive_session()— end-of-session memory storage
    │                          get_status()     — health check
    │
    ├── sessions.py         ── RECENT SESSIONS LIST.
    │                          Flat JSON file store (rag_storage/sessions.json).
    │                          Powers the "Recent sessions" sidebar in the UI.
    │
    ├── api.py              ── HTTP SERVER. THE BRIDGE.
    │                          FastAPI app. Run this to start the backend.
    │                            python api.py
    │                          Endpoints:
    │                            POST /api/chat       — main Q&A
    │                            GET  /api/sessions   — sidebar list
    │                            POST /api/archive    — save session to memory
    │                            GET  /api/documents  — book picker list
    │                            GET  /api/health     — status check
    │                            GET  /images/**      — serve extracted figures
    │
    └── embed_campbell.py   ── ONE-TIME SETUP SCRIPT.
                               Never runs during normal use. Run it once to:
                                 1. Extract figures from the PDF → backend/images/
                                 2. Build images/manifest.json (figure ID → file path)
                                 3. Chunk text with rich metadata (unit, chapter, concept, page)
                                 4. Embed chunks with Cohere → upload to Pinecone
                                 5. Upload text records to Pinecone Inference (Llama)
                               Usage:
                                 python embed_campbell.py --pdf Campbell_Biology_12e.pdf --mode both
```

---

## How a Student Question Flows Through the System

```
  ┌────────────────────────────────────────────────────────┐
  │  FRONTEND  Gnosis.html                                  │
  │                                                         │
  │  Student types → send() → fetch POST /api/chat         │
  │    { question, history, session_id, socratic, ns }      │
  └────────────────────────────┬───────────────────────────┘
                               │ HTTP
                               ▼
  ┌────────────────────────────────────────────────────────┐
  │  api.py  — validates request shape (Pydantic)           │
  └────────────────────────────┬───────────────────────────┘
                               │
                               ▼
  ┌────────────────────────────────────────────────────────┐
  │  main.py  orchestrator.chat()                           │
  │                                                         │
  │  Step 1 ─ context.check_cache(question)                 │
  │             → embed question  [models.py]               │
  │             → search cache    [storage.py / ChromaDB]   │
  │             HIT  → return instantly, skip steps 2-3     │
  │             MISS → continue                             │
  │                                                         │
  │  Step 2 ─ context.fetch_all_context(question)           │
  │             COHERE MODE:                                 │
  │               embed_query() → Cohere API                │
  │               index.query(vector) → Pinecone            │
  │             LLAMA MODE:                                  │
  │               index.search(text) → Pinecone Inference   │
  │             + retrieve_memories() → ChromaDB            │
  │             Returns: chunks + citations + image_files   │
  │                     + past session summaries            │
  │                                                         │
  │  Step 3 ─ pipeline.run_rag(context, mode)               │
  │             Selects GNOSIS_PROMPT or SOCRATIC_PROMPT    │
  │             Calls Claude / Command-R+ / Llama            │
  │             Returns: answer string                      │
  │                                                         │
  │  Step 4 ─ storage.cache_answer()  → ChromaDB            │
  │  Step 5 ─ sessions.upsert_session() → sessions.json     │
  └────────────────────────────┬───────────────────────────┘
                               │
                               ▼ JSON
  ┌────────────────────────────────────────────────────────┐
  │  FRONTEND  receives response                            │
  │                                                         │
  │  aiBubble(data)                                         │
  │    → typewriter animation on response text              │
  │    → renders source citation cards                      │
  │    → renders figure thumbnails  (click → lightbox)     │
  │    → updates Recent sessions sidebar                   │
  └────────────────────────────────────────────────────────┘


  SESSION END (student closes tab):
    beforeunload → POST /api/archive
    → pipeline.summarise_transcript()  — LLM compresses chat
    → storage.store_memory()           — saves to ChromaDB
    → sessions.mark_archived()         — updates sessions.json
    → Next session: AI knows past topics
```

---

## Common Tasks — Which File to Edit

| Task | File |
|---|---|
| Change AI tone / teaching style | `prompts.py` |
| Switch LLM (Claude ↔ Cohere ↔ Ollama) | `config.py` → `LLM_PROVIDER` |
| Switch embedder (Cohere ↔ Llama) | `config.py` → `EMBEDDER_MODE` |
| Add a new textbook | Run `embed_campbell.py` with new PDF + add entry in `pinecone_db.py → AVAILABLE_DOCUMENTS` |
| Retrieve more/fewer chunks | `config.py` → `RETRIEVAL_CONFIG.top_k_chunks` |
| Make cache more/less aggressive | `config.py` → `RETRIEVAL_CONFIG.cache_similarity_threshold` |
| Debug — see retrieved chunks in terminal | `config.py` → `VERBOSE_OUTPUT = True` |
| Clear stale cache | `rm -rf backend/rag_storage/chroma_db/` then restart |
| Change UI design | `frontend/Gnosis.html` |
| Change API port | `config.py` → `API_PORT`, and update `const API = ...` in `Gnosis.html` |

---

## Starting the App

```bash
# Terminal — backend
cd gnosis_by_zoe/backend
source venv/bin/activate
python api.py

# Browser — open the frontend
open frontend/Gnosis.html
```
