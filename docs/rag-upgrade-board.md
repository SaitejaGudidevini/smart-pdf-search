# EDMS RAG Upgrade Board

> Tracking all discussions and tasks for upgrading EDMS Smart Search to industry-level RAG on top of Mayan EDMS.

| Status Legend |
|---|
| `TODO` - Not started |
| `IN PROGRESS` - Currently working on |
| `DONE` - Completed |
| `BLOCKED` - Waiting on something |

---

# Sprint 1 — Research (All DONE)

## EDMS-001: Understand Current Codebase Architecture `DONE`
Mapped the entire codebase: `app.py` (FastAPI), `pdf_processor.py` (PyMuPDF), `search_engine.py` (fastembed + numpy), `static/index.html` (dark SPA).

## EDMS-002: Understand Current Text Extraction Flow `DONE`
Traced extraction: PyMuPDF → every line with bbox → page-level + chunk-level embeddings → numpy in-memory. Lost on restart.

## EDMS-003: Research Industry-Level RAG Architecture `DONE`
Gap analysis: current vs industry (Qdrant, BM25 hybrid, reranking, structure-aware chunking, parent-child).

## EDMS-004: Research LangChain Chunking Strategies `DONE`
Traced LangChain source code: RecursiveCharacterTextSplitter algorithm, `_merge_splits` overlap, MarkdownHeaderTextSplitter metadata.

## EDMS-013: Research Document-Specific Chunking `DONE`
Research papers → section-aware. Contracts → clause-preserving. Images → vision LLM captioning.

## EDMS-014: Research Auto-Detecting Document Type `DONE`
80% heuristics (font size, patterns), 20% LLM escalation. Document Intelligence Layer → Strategy Selector → Chunk.

## EDMS-015: Research Competitive Landscape `DONE`
NotebookLM, ChatDOC, PDF AI, Morphik. No one built full RAG chatbot on Mayan EDMS.

## EDMS-016: Research Mayan EDMS Integration `DONE`
Mayan provides: OCR, versioning, ACL, events, REST API, workflows. RAG sidecar architecture designed.

## EDMS-017: Research Orchestration Framework `DONE`
Chose: Mayan Workflow → FastAPI → Celery + Redis → LlamaIndex → Qdrant.

---

# Sprint 2 — Core RAG Pipeline (All DONE)

## EDMS-005: Structure-Aware Chunking `DONE`
**Files:** `pdf_processor.py` (enhanced), `chunking_pipeline.py` (new)
- Added `extract_structure()`: font-size analysis → heading detection (level 1/2/3) → section grouping
- Added `detect_doc_type()`: heuristic rules (academic keywords → research_paper, legal patterns → contract, images → _with_images)
- `ChunkingPipeline` routes to 3 strategies: `_chunk_research_paper`, `_chunk_contract`, `_chunk_general`
- Section prefixes: `[Section: Introduction]`, `[Clause: Article 3]`
- Skips References section (low retrieval value)

## EDMS-006: Parent-Child Chunk Strategy `DONE`
**Files:** `chunking_pipeline.py`
- Parent chunks: full section up to 4000 chars (for LLM context)
- Child chunks: 512 chars with 50 overlap via RecursiveCharacterTextSplitter (for vector search)
- Linked via `parent_id` metadata
- Search hits child → retrieves parent for RAG context

## EDMS-007: Qdrant Vector Store `DONE`
**Files:** `search_engine.py` (rewritten)
- Replaced numpy arrays with Qdrant (in-memory mode, HNSW indexed, cosine distance)
- Full metadata payload per chunk (section, page, chunk_type, parent_id, lines with bbox)
- `get_parent_chunk(parent_id)` for RAG context expansion
- Batch upsert (100 per batch)

## EDMS-008: BM25 Hybrid Search `DONE`
**Files:** `search_engine.py`
- Added `rank_bm25.BM25Okapi` keyword index built from child chunks
- Parallel search: Qdrant vector (top 50) + BM25 keyword (top 50)
- Reciprocal Rank Fusion: `RRF_score(d) = Σ(1 / (60 + rank_i))`
- Documents appearing in both systems get boosted scores

## EDMS-009: Cross-Encoder Reranking `DONE`
**Files:** `search_engine.py`
- Optional `sentence-transformers` CrossEncoder (`ms-marco-MiniLM-L-6-v2`)
- Reranks top 15 fused results if installed
- Graceful fallback if not installed

## EDMS-010: Improved RAG Prompt `DONE`
**Files:** `app.py`
- Strict 5-rule grounding prompt: "Use ONLY excerpts, cite [Page X], say if not enough info"
- Parent chunks used as LLM context (richer than raw page text)
- Groq (llama-3.3-70b) as primary LLM provider with temperature=0

## EDMS-018: Document Type Auto-Detection `DONE`
**Files:** `pdf_processor.py`
- Font profile analysis: Counter of all font sizes → body_size = most common
- Heading classification: >1.3x body = level 1, >1.1x body + bold = level 2
- Doc type heuristics: 3+ academic keywords → research_paper, legal patterns → contract
- Image block detection → appends `_with_images`

---

# Sprint 3 — Mayan EDMS Integration (All DONE)

## EDMS-020: Mayan EDMS Docker Setup `DONE`
**Files:** `docker/docker-compose.yml`, `docker/.env`
- Full Docker Compose: Mayan (app + PostgreSQL + RabbitMQ + Redis) + RAG sidecar
- Mayan on port 8000, sidecar on port 8080 (internal + exposed)
- Persistent volumes for media, postgres data, uploads, cache

## EDMS-021: Mayan REST API Bridge `DONE`
**Files:** `src/smart-search/mayan_bridge.py` (new)
- Token-based auth with Mayan API v4
- `list_documents()`, `get_document()`, `get_document_file()` (download via files endpoint)
- `get_ocr_text()` per page, `get_document_metadata()` (tags, cabinets)
- `sync_document()` — full download + metadata in one call

## EDMS-022: Mayan Sync Endpoints `DONE`
**Files:** `src/smart-search/app.py`
- `GET /api/mayan/status` — check Mayan connectivity
- `GET /api/mayan/documents` — list all Mayan documents
- `POST /api/mayan/sync/{doc_id}` — download → extract → chunk → index with Mayan metadata
- `POST /api/mayan/webhook` — auto-index on Mayan events
- Enriches chunks with `mayan_doc_id`, `mayan_tags`, `mayan_cabinet`

## EDMS-023: RAG Chat as Native Mayan Django App `DONE`
**Files:** `src/mayan_rag_chat/` (new Django app)
- `apps.py` — registers as `MayanAppConfig` with `app_url='rag_chat'`
- `views.py` — ChatView, IndexAllView, SearchAPIView, IndexDocumentView
- `urls.py` — `/rag_chat/chat/`, `/rag_chat/index/`, `/rag_chat/api/search/`
- Proxies to RAG sidecar (heavy ML runs in separate container, avoids OOM in Mayan)
- Templates extend Mayan's `appearance/base.html` (native look and feel)

## EDMS-024: Chat UI with PDF Viewer `DONE`
**Files:** `src/mayan_rag_chat/templates/rag_chat/chat.html`
- Split layout: chat panel (left) + floating PDF viewer (right)
- AI answers with clickable `[Page X]` citation badges
- Source cards with keyword highlighting and match scores
- Click source → opens PDF page with bbox-based yellow highlights
- Navigation buttons: Chat, Index Documents, All Documents
- Prev/Next page navigation in viewer

## EDMS-025: Groq LLM Integration `DONE`
**Files:** `app.py`
- Added `_call_groq()` provider (llama-3.3-70b-versatile, temperature=0)
- Priority: Groq → Ollama → Anthropic → OpenAI
- CORS middleware for cross-origin PDF viewer requests

## EDMS-026: CORS + Multi-Model Support `DONE`
**Files:** `app.py`
- FastAPI CORSMiddleware (allow all origins for Mayan → sidecar communication)
- Switched to `all-MiniLM-L6-v2` embedding model (lighter, runs in Docker without OOM)

---

# Sprint 4 — Upcoming Improvements

## EDMS-011: Upgrade Embedding Model `TODO`
Benchmark `bge-large-en-v1.5` (1024 dim) vs current `all-MiniLM-L6-v2` (384 dim). Measure accuracy/memory tradeoff.

## EDMS-012: Multi-Document Search `TODO`
Currently re-indexes on each sync (replaces previous). Need: append to index, filter by document, search across all docs simultaneously.

## EDMS-019: Multimodal Image Captioning `TODO`
Extract PDF images → caption with vision LLM (Claude/GPT-4V) → store as searchable chunks.

## EDMS-027: Auto-Index on Upload `TODO`
Configure Mayan workflow to POST to `/api/mayan/webhook` when OCR completes. Zero-touch indexing.

## EDMS-028: Fix PDF Highlighting in Mayan Chat `TODO`
Bounding box highlights work (verified server-side) but browser may have CORS/rendering issues. Debug and fix.

## EDMS-029: Persistent Vector Store `TODO`
Replace in-memory Qdrant with Qdrant server (separate container). Survives sidecar restarts without re-indexing.

## EDMS-030: Chat History & Sessions `TODO`
Save chat conversations per user. Resume previous sessions. Export chat as PDF.

## EDMS-031: Document-Scoped Search `TODO`
Add dropdown in chat UI to filter search to a specific document or search across all.

## EDMS-032: Improved UI/UX `TODO`
- Show actual chunk/document stats (query sidecar for real counts)
- Loading spinner during indexing
- Better error messages
- Mobile responsive layout

---

# Architecture Summary

```
┌─────────────────────────────────────────────────┐
│  MAYAN EDMS (port 8000)                         │
│  Upload → OCR → Store → /rag_chat/ Django App   │
│  Login: admin / admin123                        │
└──────────────┬──────────────────────────────────┘
               │ proxies via httpx
               ▼
┌─────────────────────────────────────────────────┐
│  RAG SIDECAR (port 8080)                        │
│  FastAPI + PyMuPDF + fastembed + Qdrant + BM25  │
│                                                  │
│  /api/mayan/sync/{id}  ← index from Mayan      │
│  /api/search           ← hybrid search + RAG    │
│  /api/page/{n}/highlighted ← bbox highlights    │
│                                                  │
│  Pipeline:                                       │
│  PDF → extract_structure (font analysis)         │
│      → detect_doc_type (research/contract/gen)   │
│      → chunk_document (parent-child, section)    │
│      → index (Qdrant vectors + BM25 keywords)    │
│      → search (vector + BM25 + RRF fusion)       │
│      → Groq LLM answer with [Page X] citations  │
└─────────────────────────────────────────────────┘

Tech Stack:
  Mayan EDMS    — Django, PostgreSQL, RabbitMQ, Redis, Tesseract OCR
  RAG Sidecar   — FastAPI, PyMuPDF, fastembed (all-MiniLM-L6-v2), Qdrant, rank_bm25
  LLM           — Groq (llama-3.3-70b) → Ollama → Anthropic → OpenAI
  Deployment    — Docker Compose (5 containers)
```

---

# File Map

```
EDMS/
├── docker/
│   ├── docker-compose.yml          # Full stack: Mayan + RAG sidecar
│   └── .env                        # GROQ_API_KEY
├── docs/
│   ├── rag-upgrade-board.md        # This file
│   └── system-architecture.md      # Original architecture doc
├── src/
│   ├── smart-search/               # RAG Sidecar Service
│   │   ├── app.py                  # FastAPI server + Mayan endpoints + LLM providers
│   │   ├── pdf_processor.py        # PyMuPDF extraction + structure detection + highlighting
│   │   ├── chunking_pipeline.py    # Strategy routing + parent-child chunking
│   │   ├── search_engine.py        # Qdrant + BM25 + RRF hybrid search
│   │   ├── mayan_bridge.py         # Mayan REST API client
│   │   ├── static/index.html       # Standalone dark-theme UI
│   │   ├── Dockerfile              # Container build
│   │   └── requirements.txt        # Python deps
│   └── mayan_rag_chat/             # Mayan Django App (plugin)
│       ├── apps.py                 # MayanAppConfig registration
│       ├── views.py                # Chat, Index, Search views (proxy to sidecar)
│       ├── urls.py                 # /rag_chat/ URL routing
│       ├── rag_engine.py           # In-process RAG engine (unused — too heavy for Mayan)
│       ├── templates/rag_chat/
│       │   ├── chat.html           # Chatbot UI with PDF viewer panel
│       │   └── index_all.html      # Document indexing page
│       └── migrations/
```
