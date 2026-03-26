"""Smart Search Tool: Source-First PDF search with highlighted screenshots."""

import os
import json
import httpx
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, Response, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

load_dotenv(override=True)

from pdf_processor import PDFProcessor
from search_engine import SearchEngine
from chunking_pipeline import ChunkingPipeline
from mayan_bridge import MayanBridge
from storage_pg import make_document_key
from excel_parser import ExcelParser, is_excel_file
from excel_enricher import ExcelEnricher
from excel_storage import ExcelStorageManager, classify_query
from excel_agent import ExcelSQLAgent
from excel_routes import router as excel_router

# Structure extraction: pymupdf4llm (best) → Docling (ML-based) → PyMuPDF (basic fallback)
_structure_parser = None
_parser_name = "pymupdf"

try:
    from pymupdf4llm_parser import PyMuPDF4LLMParser
    _structure_parser = PyMuPDF4LLMParser()
    _parser_name = "pymupdf4llm"
    print("Using pymupdf4llm for document structure extraction (dynamic heading hierarchy).")
except ImportError:
    try:
        from docling_parser import DoclingParser
        _structure_parser = DoclingParser()
        _parser_name = "docling"
        print("Using Docling for document structure extraction.")
    except ImportError:
        print("Falling back to PyMuPDF for structure extraction.")

app = FastAPI(title="Smart PDF Search")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
pdf_processor = PDFProcessor(cache_dir="cache")  # kept for rendering + page extraction
search_engine = SearchEngine()
chunking_pipeline = ChunkingPipeline()
mayan = MayanBridge()
excel_parser = ExcelParser()
excel_enricher = ExcelEnricher()
excel_storage = ExcelStorageManager()
excel_agent = ExcelSQLAgent()


def extract_structure(pdf_path: str):
    """Extract document structure: pymupdf4llm → Docling → PyMuPDF fallback."""
    if _structure_parser is not None:
        try:
            return _structure_parser.extract_structure(pdf_path)
        except Exception as e:
            print(f"{_parser_name} failed, falling back to PyMuPDF: {e}")
    return pdf_processor.extract_structure(pdf_path)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Track the currently loaded PDF
current_pdf: dict = {"path": None, "name": None, "page_count": 0}

app.mount("/static", StaticFiles(directory="static"), name="static")
app.include_router(excel_router)  # /api/excel/upload, /api/excel/status, /api/excel/retry


@app.get("/", response_class=HTMLResponse)
async def index():
    return Path("static/index.html").read_text()


@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF or Excel file and index it for searching."""
    filename = file.filename.lower()

    if not (filename.endswith(".pdf") or is_excel_file(filename)):
        raise HTTPException(400, "Supported formats: .pdf, .xlsx, .xls, .csv")

    file_path = UPLOAD_DIR / file.filename
    content = await file.read()
    file_path.write_bytes(content)

    if is_excel_file(filename):
        return _index_excel(str(file_path), file.filename)
    else:
        return _index_pdf(str(file_path), file.filename)


def _index_pdf(pdf_path: str, filename: str) -> dict:
    """Index a PDF file through the existing pipeline."""
    pages = pdf_processor.extract_pages(pdf_path)
    page_count = pdf_processor.get_page_count(pdf_path)

    structure = extract_structure(pdf_path)
    doc_type = structure.doc_type
    document_key = make_document_key(filename)
    chunks = chunking_pipeline.chunk_document(structure, filename, document_key=document_key)
    search_engine.index(chunks, pages, document_name=filename)

    current_pdf["path"] = pdf_path
    current_pdf["name"] = filename
    current_pdf["page_count"] = page_count

    return {
        "filename": filename,
        "page_count": page_count,
        "indexed_pages": len(pages),
        "total_lines": sum(len(p.lines) for p in pages),
        "doc_type": doc_type,
        "total_chunks": len(chunks),
    }


def _index_excel(file_path: str, filename: str, mayan_doc_id: int | None = None) -> dict:
    """Index an Excel file: parse → enrich → chunk → embed → store."""
    # Stage 1: Adaptive parsing
    dataframes, formulas = excel_parser.parse(file_path)

    # Stage 2: Structural enrichment — semantic rows for embedding
    all_semantic_rows = []
    for sheet_name, df in dataframes.items():
        sheet_formulas = formulas.get(sheet_name, [])
        rows = excel_enricher.generate_semantic_rows(df, sheet_name, sheet_formulas)
        all_semantic_rows.extend(rows)

    # Generate schema description (stored as metadata for future SQL agent)
    schema_description = excel_enricher.generate_all_schemas(dataframes)

    # Convert to DocumentStructure so existing chunking pipeline works
    structure = excel_parser.extract_structure(file_path)
    document_key = make_document_key(filename, mayan_doc_id)
    chunks = chunking_pipeline.chunk_document(structure, filename, document_key=document_key)

    # Enrich all chunks with Excel-specific metadata
    for chunk in chunks:
        chunk.metadata["file_type"] = "spreadsheet"
        chunk.metadata["schema_description"] = schema_description
        chunk.metadata["sheet_names"] = list(dataframes.keys())
        chunk.metadata["total_rows"] = sum(len(df) for df in dataframes.values())
        if mayan_doc_id:
            chunk.metadata["mayan_doc_id"] = mayan_doc_id

    # Index into pgvector (same as PDF path)
    search_engine.index(chunks, pages=None, document_id=mayan_doc_id, document_name=filename)

    # Stage 3: Store DataFrames in SQLite for SQL queries
    has_sql = False
    if dataframes:
        try:
            excel_storage.store_excel(document_key, filename, dataframes)
            has_sql = True
        except Exception as e:
            print(f"[Excel] SQLite storage failed (non-fatal): {e}")

    return {
        "filename": filename,
        "document_key": document_key,
        "file_type": "spreadsheet",
        "sheets": {name: {"rows": len(df), "columns": len(df.columns)} for name, df in dataframes.items()},
        "total_rows": sum(len(df) for df in dataframes.values()),
        "total_semantic_rows": len(all_semantic_rows),
        "total_chunks": len(chunks),
        "doc_type": "spreadsheet",
        "schema_description": schema_description,
        "sql_queryable": has_sql,
    }


@app.get("/api/page/{page_number}")
async def get_page_image(page_number: int):
    """Get a clean page image (no highlights)."""
    if not current_pdf["path"]:
        raise HTTPException(400, "No PDF uploaded")
    if page_number < 1 or page_number > current_pdf["page_count"]:
        raise HTTPException(400, f"Page must be between 1 and {current_pdf['page_count']}")

    img = pdf_processor.get_page_image(current_pdf["path"], page_number)
    return Response(content=img, media_type="image/png")


@app.post("/api/search")
async def search(body: dict):
    """Search the PDF and return results with highlighted page screenshots."""
    query = body.get("query", "").strip()
    if not query:
        raise HTTPException(400, "Query is required")
    if not current_pdf["path"] and not body.get("document_id") and not search_engine.has_documents():
        raise HTTPException(400, "No PDF uploaded")

    document_id = body.get("document_id")
    results = search_engine.search(query, top_k=5, document_id=document_id)

    response_results = []
    for result in results:
        chunks_response = []
        highlight_bboxes = []

        for chunk in result.matched_chunks:
            # Collect bounding boxes for direct highlighting (no text re-search)
            for line in chunk.lines:
                if line.get("bbox"):
                    highlight_bboxes.append(line["bbox"])

            chunks_response.append({
                "text": chunk.text,
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "score": round(chunk.score, 3),
                "match_type": chunk.match_type,
                "line_count": len(chunk.lines),
            })

        response_results.append({
            "document_id": result.document_id,
            "document_name": result.document_name,
            "document_key": result.document_key,
            "page_number": result.page_number,
            "page_score": round(result.page_score, 3),
            "chunks": chunks_response,
            "highlight_bboxes": highlight_bboxes,
        })

    # Build RAG context: for each matched child, extract a context window
    # from its parent centered around the child's location.
    RAG_CONTEXT_BUDGET = 8000  # chars per context block — larger for financial tables
    rag_context = []
    seen_parents = set()
    for result in results[:3]:
        for chunk in result.matched_chunks:
            parent_id = chunk.metadata.get("parent_id") if hasattr(chunk, "metadata") and chunk.metadata else None
            if parent_id and parent_id in seen_parents:
                continue
            if parent_id:
                seen_parents.add(parent_id)

            parent_text = None
            if parent_id:
                parent_text = search_engine.get_parent_chunk(parent_id)

            if parent_text and len(parent_text) <= RAG_CONTEXT_BUDGET:
                # Parent fits in budget — use it whole
                context_text = parent_text
            elif parent_text:
                # Parent too large — extract a window centered on the child match
                context_text = _extract_context_window(
                    parent_text, chunk.text, RAG_CONTEXT_BUDGET
                )
            else:
                context_text = chunk.text

            rag_context.append({
                "document_name": result.document_name,
                "page_number": result.page_number,
                "text": context_text,
            })
            if len(rag_context) >= 5:
                break
        if len(rag_context) >= 3:
            break

    # Get AI answer via RAG
    ai_summary = None
    if rag_context:
        ai_summary = await _generate_rag_answer(query, rag_context)

    return {
        "query": query,
        "topic": search_engine.extract_topic(query),
        "total_results": len(response_results),
        "results": response_results,
        "ai_summary": ai_summary,
        "pdf_name": current_pdf["name"] or (results[0].document_name if results else None),
    }


@app.get("/api/page/{page_number}/highlighted")
async def get_highlighted_page(page_number: int, texts: str = ""):
    """Get a page image with specific texts highlighted."""
    if not current_pdf["path"]:
        raise HTTPException(400, "No PDF uploaded")

    highlight_texts = json.loads(texts) if texts else []

    if highlight_texts:
        img = pdf_processor.get_highlighted_page_image(
            current_pdf["path"], page_number, highlight_texts
        )
    else:
        img = pdf_processor.get_page_image(current_pdf["path"], page_number)

    return Response(content=img, media_type="image/png")


@app.post("/api/page/{page_number}/highlighted")
async def get_highlighted_page_post(page_number: int, body: dict):
    """Get a page image with bounding box regions highlighted."""
    if not current_pdf["path"]:
        raise HTTPException(400, "No PDF uploaded")

    bboxes = body.get("bboxes", [])
    texts = body.get("texts", [])

    if bboxes:
        # Use exact bounding boxes (reliable)
        img = pdf_processor.get_highlighted_page_with_bboxes(
            current_pdf["path"], page_number, bboxes
        )
    elif texts:
        # Fallback to text search
        img = pdf_processor.get_highlighted_page_image(
            current_pdf["path"], page_number, texts
        )
    else:
        img = pdf_processor.get_page_image(current_pdf["path"], page_number)

    return Response(content=img, media_type="image/png")


def _extract_context_window(parent_text: str, child_text: str, budget: int) -> str:
    """Extract a window from parent_text centered on where child_text appears.

    If the child text is found in the parent, the window is centered around it
    so the LLM sees the matched content plus surrounding context.  If not found
    (child was split differently), fall back to the beginning of the parent.
    """
    # Strip section prefix from child for matching
    child_core = child_text
    if child_core.startswith("[Section:"):
        idx = child_core.find("] ")
        if idx != -1:
            child_core = child_core[idx + 2:]
    if child_core.startswith("[Clause:"):
        idx = child_core.find("] ")
        if idx != -1:
            child_core = child_core[idx + 2:]

    # Find the child's location within the parent
    match_start = parent_text.find(child_core[:80])
    if match_start == -1:
        # Not found — return the beginning of the parent
        return parent_text[:budget]

    # Center a window of `budget` chars around the match
    half = budget // 2
    window_start = max(0, match_start - half)
    window_end = min(len(parent_text), window_start + budget)
    # Adjust start if we hit the end
    if window_end - window_start < budget:
        window_start = max(0, window_end - budget)

    window = parent_text[window_start:window_end]

    # Add ellipsis markers if truncated
    prefix = "..." if window_start > 0 else ""
    suffix = "..." if window_end < len(parent_text) else ""
    return f"{prefix}{window}{suffix}"


async def _generate_rag_answer(query: str, rag_context: list[dict]) -> dict:
    """RAG: Send page context to LLM. Tries Ollama (free) → Anthropic → OpenAI."""
    sources_text = ""
    for ctx in rag_context:
        sources_text += f"\n\n--- PAGE {ctx['page_number']} ---\n{ctx['text']}"

    if not sources_text.strip():
        return None

    system_prompt = (
        "You are a financial data analyst. Answer questions using the provided document excerpts. "
        "Rules:\n"
        "1. Use information from the provided excerpts as your primary source.\n"
        "2. Cite sources as [Page X] for every claim.\n"
        "3. COMPUTE, CALCULATE, and REASON when asked. If the user asks for totals, "
        "differences, percentages, or comparisons — do the math using the numbers in the excerpts.\n"
        "4. Show your calculations step by step when performing math.\n"
        "5. Present numerical answers clearly with proper formatting ($, commas, %).\n"
        "6. If data is available across multiple pages, combine it to give a complete answer.\n"
        "7. If the excerpts truly don't contain the needed data, say so — but try to answer "
        "with what's available before giving up."
    )

    user_prompt = f"DOCUMENT PAGES:{sources_text}\n\nQUESTION: {query}"

    # Priority: Groq (fast) → Ollama (free) → Anthropic → OpenAI
    providers = [
        ("Groq (llama-3.3-70b)", _call_groq),
        ("Ollama (llama3.2)", _call_ollama),
        ("Claude Haiku 4.5", _call_anthropic),
        ("GPT-4o Mini", _call_openai),
    ]

    for model_name, call_fn in providers:
        try:
            text = await call_fn(system_prompt, user_prompt)
            if text:
                return {
                    "text": text,
                    "disclaimer": f"AI answer by {model_name}. Verify against the highlighted source.",
                    "enabled": True,
                    "model": model_name
                }
        except Exception:
            continue  # try next provider

    return None


async def _call_groq(system_prompt: str, user_prompt: str) -> str | None:
    """Call Groq API (needs GROQ_API_KEY). Fast inference with llama-3.3-70b."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": 500,
                "temperature": 0,
            }
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        return data["choices"][0]["message"]["content"]


async def _call_ollama(system_prompt: str, user_prompt: str) -> str | None:
    """Call local Ollama (free, no API key needed)."""
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.post(
            "http://localhost:11434/api/chat",
            json={
                "model": "llama3.2",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
            }
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        return data.get("message", {}).get("content")


async def _call_anthropic(system_prompt: str, user_prompt: str) -> str | None:
    """Call Anthropic API (needs ANTHROPIC_API_KEY)."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": "claude-haiku-4-5-20251001",
                "max_tokens": 500,
                "system": system_prompt,
                "messages": [{"role": "user", "content": user_prompt}]
            }
        )
        data = resp.json()
        return data.get("content", [{}])[0].get("text")


async def _call_openai(system_prompt: str, user_prompt: str) -> str | None:
    """Call OpenAI API (needs OPENAI_API_KEY)."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "gpt-4o-mini",
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "max_tokens": 500
            }
        )
        data = resp.json()
        return data["choices"][0]["message"]["content"]


# ============================================================
# Mayan EDMS Integration Endpoints
# ============================================================

@app.get("/api/mayan/documents")
async def mayan_list_documents():
    """List all documents from Mayan EDMS."""
    try:
        docs = await mayan.list_documents()
        return {"documents": docs, "count": len(docs)}
    except Exception as e:
        raise HTTPException(502, f"Failed to connect to Mayan EDMS: {e}")


@app.post("/api/mayan/sync/{doc_id}")
async def mayan_sync_document(doc_id: int):
    """Sync a document from Mayan EDMS: download → extract → chunk → index → classify back."""
    try:
        # 1. Download from Mayan
        mayan_doc = await mayan.sync_document(doc_id)

        # Route: Excel files get their own pipeline
        if is_excel_file(mayan_doc.file_path):
            result = _index_excel(mayan_doc.file_path, mayan_doc.label, mayan_doc_id=doc_id)
            await mayan.classify_document(doc_id=doc_id, rag_doc_type="spreadsheet", tags=["AI: Spreadsheet"])
            result["mayan_doc_id"] = doc_id
            result["mayan_metadata"] = mayan_doc.metadata
            return result

        ocr_pages = await mayan.get_ocr_text(doc_id)

        # 2. Extract pages (for frontend viewer)
        pages = pdf_processor.extract_pages(mayan_doc.file_path)
        page_count = pdf_processor.get_page_count(mayan_doc.file_path)

        # 3. Check if Mayan already has a user-assigned type (not "Default")
        #    If so, use it directly instead of calling Groq
        mayan_type = mayan_doc.metadata.get("document_type", "Default")
        hint_doc_type = _mayan_type_to_rag_type(mayan_type)

        # 4. Structure-aware extraction + chunking
        structure = extract_structure(mayan_doc.file_path)

        # If Mayan had a real type, override whatever the parser detected
        if hint_doc_type:
            has_images = "_with_images" in structure.doc_type
            structure.doc_type = hint_doc_type + ("_with_images" if has_images else "")
            print(f"Using Mayan document type '{mayan_type}' → {structure.doc_type} (skipped Groq)")

        pages, structure, source_profile = pdf_processor.apply_ocr_text(pages, structure, ocr_pages)
        doc_type = structure.doc_type
        document_key = make_document_key(mayan_doc.label, doc_id)
        chunks = chunking_pipeline.chunk_document(structure, mayan_doc.label, document_key=document_key)

        # Enrich chunks with Mayan metadata
        for chunk in chunks:
            chunk.metadata["mayan_doc_id"] = doc_id
            chunk.metadata["mayan_tags"] = mayan_doc.metadata.get("tags", [])
            chunk.metadata["mayan_cabinet"] = mayan_doc.metadata.get("cabinets", [])
            chunk.metadata["text_source_mode"] = source_profile.get("mode")

        # 5. Index
        search_engine.index(chunks, pages, document_id=doc_id, document_name=mayan_doc.label)

        # 6. Write classification back to Mayan (type + tags)
        #    Only writes if current type is "Default" (user hasn't classified it)
        classify_result = await mayan.classify_document(
            doc_id=doc_id,
            rag_doc_type=doc_type,
            tags=[_rag_type_to_tag(doc_type)],
        )

        # 7. Extract metadata via Groq and write to Mayan
        text_preview = ""
        for p in structure.pages[:3]:
            for s in p.sections:
                if s.section_type not in ("image", "table") and s.level == 3:
                    text_preview += s.content + "\n"
                    if len(text_preview) > 2000:
                        break
            if len(text_preview) > 2000:
                break

        metadata_result = await mayan.extract_and_write_metadata(
            doc_id=doc_id,
            rag_doc_type=doc_type,
            text_preview=text_preview[:3000],
        )

        # Update current_pdf for the viewer
        current_pdf["path"] = mayan_doc.file_path
        current_pdf["name"] = mayan_doc.label
        current_pdf["page_count"] = page_count

        return {
            "mayan_doc_id": doc_id,
            "filename": mayan_doc.label,
            "page_count": page_count,
            "doc_type": doc_type,
            "total_chunks": len(chunks),
            "source_profile": source_profile,
            "mayan_metadata": mayan_doc.metadata,
            "classification": classify_result,
            "extracted_metadata": metadata_result,
        }
    except Exception as e:
        raise HTTPException(502, f"Failed to sync document {doc_id}: {e}")


# Map Mayan DocumentType label → RAG doc_type (for user-assigned types)
_MAYAN_TO_RAG = {
    "Research Paper": "research_paper",
    "Contract": "contract",
    "Invoice": "invoice",
    "Resume": "resume",
    "Technical Manual": "technical_manual",
    "Financial Report": "financial_report",
    "Medical Document": "medical_document",
    "Presentation": "presentation",
    "Spreadsheet": "spreadsheet",
}


def _mayan_type_to_rag_type(mayan_label: str) -> str | None:
    """Convert Mayan DocumentType label to RAG doc_type. Returns None for 'Default'."""
    return _MAYAN_TO_RAG.get(mayan_label)


def _rag_type_to_tag(doc_type: str) -> str:
    """Convert RAG doc_type to a human-readable tag label."""
    base = doc_type.replace("_with_images", "").replace("_", " ").title()
    return f"AI: {base}"


@app.get("/api/mayan/chunks/{doc_id}")
async def mayan_document_chunks(doc_id: int, limit: int = 100):
    """Return stored parent and child chunks for a Mayan document."""
    if limit < 1 or limit > 500:
        raise HTTPException(400, "limit must be between 1 and 500")

    document_key = make_document_key(str(doc_id), doc_id)
    payload = search_engine.store.get_document_chunks(
        document_key=document_key,
        limit_per_type=limit,
    )
    if not payload:
        raise HTTPException(404, "No chunks found for document")

    return payload


@app.post("/api/mayan/webhook")
async def mayan_webhook(body: dict):
    """Webhook endpoint for Mayan EDMS events (document uploaded, OCR complete).

    Configure Mayan workflow to POST to this endpoint when documents are ready.
    Payload: {"document_id": 123, "event": "document.ocr.complete"}
    """
    doc_id = body.get("document_id")
    event = body.get("event", "unknown")

    if not doc_id:
        raise HTTPException(400, "document_id required")

    # Auto-sync the document into our RAG pipeline
    try:
        # Set "⏳ Processing" tag immediately
        await mayan.set_status_processing(doc_id)

        mayan_doc = await mayan.sync_document(doc_id)
        document_key = make_document_key(mayan_doc.label, doc_id)
        event_key = body.get("event_id") or f"{event}:doc:{doc_id}:version:{mayan_doc.version_id or 'unknown'}"
        should_process, existing_event = search_engine.store.begin_webhook_event(
            event_key=event_key,
            event_name=event,
            document_key=document_key,
            mayan_doc_id=doc_id,
            mayan_version_id=mayan_doc.version_id,
            payload=body,
        )
        if not should_process:
            await mayan.set_status_ready(doc_id)
            return {
                "status": "duplicate",
                "event": event,
                "event_key": event_key,
                "mayan_doc_id": doc_id,
                "mayan_version_id": mayan_doc.version_id,
                "existing_status": existing_event["status"] if existing_event else "completed",
            }

        # Route: Excel files get their own pipeline
        if is_excel_file(mayan_doc.file_path):
            result = _index_excel(mayan_doc.file_path, mayan_doc.label, mayan_doc_id=doc_id)
            search_engine.store.complete_webhook_event(event_key)
            await mayan.classify_document(doc_id=doc_id, rag_doc_type="spreadsheet", tags=["AI: Spreadsheet"])
            await mayan.set_status_ready(doc_id)
            return {
                "status": "indexed",
                "event": event,
                "event_key": event_key,
                "mayan_doc_id": doc_id,
                "doc_type": "spreadsheet",
                "total_chunks": result["total_chunks"],
                "source_profile": {"mode": "excel"},
            }

        ocr_pages = await mayan.get_ocr_text(doc_id)
        pages = pdf_processor.extract_pages(mayan_doc.file_path)
        structure = extract_structure(mayan_doc.file_path)

        # Use Mayan type if not "Default"
        mayan_type = mayan_doc.metadata.get("document_type", "Default")
        hint_doc_type = _mayan_type_to_rag_type(mayan_type)
        if hint_doc_type:
            has_images = "_with_images" in structure.doc_type
            structure.doc_type = hint_doc_type + ("_with_images" if has_images else "")

        pages, structure, source_profile = pdf_processor.apply_ocr_text(pages, structure, ocr_pages)
        doc_type = structure.doc_type
        chunks = chunking_pipeline.chunk_document(structure, mayan_doc.label, document_key=document_key)

        for chunk in chunks:
            chunk.metadata["mayan_doc_id"] = doc_id
            chunk.metadata["mayan_version_id"] = mayan_doc.version_id
            chunk.metadata["text_source_mode"] = source_profile.get("mode")

        search_engine.index(chunks, pages, document_id=doc_id, document_name=mayan_doc.label)
        search_engine.store.complete_webhook_event(event_key)

        # Write classification + metadata back to Mayan
        await mayan.classify_document(doc_id=doc_id, rag_doc_type=doc_type, tags=[_rag_type_to_tag(doc_type)])

        text_preview = ""
        for p in structure.pages[:3]:
            for s in p.sections:
                if s.section_type not in ("image", "table") and s.level == 3:
                    text_preview += s.content + "\n"
                    if len(text_preview) > 2000:
                        break
            if len(text_preview) > 2000:
                break
        await mayan.extract_and_write_metadata(doc_id=doc_id, rag_doc_type=doc_type, text_preview=text_preview[:3000])

        # Set "✅ Indexed" tag — removes "⏳ Processing"
        await mayan.set_status_ready(doc_id)

        return {
            "status": "indexed",
            "event": event,
            "event_key": event_key,
            "mayan_doc_id": doc_id,
            "mayan_version_id": mayan_doc.version_id,
            "doc_type": doc_type,
            "total_chunks": len(chunks),
            "source_profile": source_profile,
        }
    except Exception as e:
        try:
            if 'event_key' in locals():
                search_engine.store.fail_webhook_event(event_key, str(e))
            await mayan.set_status_error(doc_id)
        except Exception:
            pass
        return {
            "status": "error",
            "event": event,
            "event_key": locals().get("event_key"),
            "mayan_doc_id": doc_id,
            "error": str(e),
        }


@app.get("/excel-chat", response_class=HTMLResponse)
async def excel_chat_ui():
    """Excel RAG Chat UI with backend engine panel."""
    return Path("static/excel_chat.html").read_text()


@app.post("/api/excel/chat")
async def excel_chat(body: dict):
    """Chat endpoint with full pipeline trace for the UI."""
    question = body.get("question", "").strip()
    document_key = body.get("document_key", "").strip()

    if not question:
        return {"error": "question is required"}
    if not document_key:
        return {"error": "document_key is required"}

    trace = {}

    # Check SQL database
    if not excel_storage.has_sql_database(document_key):
        # Fall back to semantic search
        query_type = "semantic"
        trace["note"] = "No SQL database for this document"
    else:
        query_type = classify_query(question)

    trace["query_type"] = query_type

    if query_type == "sql":
        schema = excel_storage.get_schema_description(document_key)
        trace["schema"] = schema

        result = excel_agent.ask(
            question=question,
            schema=schema,
            execute_fn=lambda sql: excel_storage.execute_sql(document_key, sql),
        )

        trace["sql"] = result.get("sql")
        trace["results"] = result.get("results", [])
        trace["attempts"] = result.get("attempts", 1)
        trace["model"] = result.get("model", "unknown")

        return {
            "query_type": "sql",
            "question": question,
            "answer": result.get("answer", result.get("error", "No answer")),
            "trace": trace,
        }
    else:
        # Semantic search — needs PostgreSQL. If not available, try SQL path instead.
        if not search_engine.store._pg_available:
            # No PostgreSQL — if we have a SQL database, route all queries through SQL
            if excel_storage.has_sql_database(document_key):
                schema = excel_storage.get_schema_description(document_key)
                trace["schema"] = schema
                trace["note"] = "PostgreSQL unavailable — routed to SQL agent"
                result = excel_agent.ask(
                    question=question,
                    schema=schema,
                    execute_fn=lambda sql: excel_storage.execute_sql(document_key, sql),
                )
                trace["sql"] = result.get("sql")
                trace["results"] = result.get("results", [])
                trace["attempts"] = result.get("attempts", 1)
                trace["model"] = result.get("model", "unknown")
                return {
                    "query_type": "sql",
                    "question": question,
                    "answer": result.get("answer", result.get("error", "No answer")),
                    "trace": trace,
                }
            return {
                "query_type": "semantic",
                "question": question,
                "answer": "Semantic search requires PostgreSQL. Please start the Docker stack or ask a data question.",
                "trace": trace,
            }

        results = search_engine.search(question, top_k=5)
        trace["results_count"] = len(results)

        rag_context = []
        for r in results[:3]:
            for chunk in r.matched_chunks:
                rag_context.append({
                    "document_name": r.document_name,
                    "page_number": r.page_number,
                    "text": chunk.text[:500],
                })

        ai_answer = None
        if rag_context:
            ai_answer = await _generate_rag_answer(question, rag_context)

        return {
            "query_type": "semantic",
            "question": question,
            "answer": ai_answer.get("text") if ai_answer else "No relevant results found.",
            "results": [
                {"document_name": r.document_name, "page_number": r.page_number, "score": round(r.page_score, 3)}
                for r in results
            ],
            "trace": trace,
        }


@app.post("/api/excel/query")
async def excel_query(body: dict):
    """Ask a natural language question about an indexed Excel document.

    Routes to SQL agent for computation queries (sum, average, count)
    or to standard search for semantic queries.
    """
    question = body.get("question", "").strip()
    document_key = body.get("document_key", "").strip()

    if not question:
        raise HTTPException(400, "question is required")
    if not document_key:
        raise HTTPException(400, "document_key is required")

    # Check if this document has a SQL database
    if not excel_storage.has_sql_database(document_key):
        raise HTTPException(404, "No SQL database found for this document. Was it indexed as Excel?")

    # Classify: does this question need SQL or semantic search?
    query_type = classify_query(question)

    if query_type == "sql":
        schema = excel_storage.get_schema_description(document_key)
        result = excel_agent.ask(
            question=question,
            schema=schema,
            execute_fn=lambda sql: excel_storage.execute_sql(document_key, sql),
        )
        return {
            "query_type": "sql",
            "question": question,
            "document_key": document_key,
            **result,
        }
    else:
        # Fall back to standard hybrid search
        results = search_engine.search(question, top_k=5, document_id=None)
        return {
            "query_type": "semantic",
            "question": question,
            "document_key": document_key,
            "total_results": len(results),
            "results": [
                {
                    "document_name": r.document_name,
                    "page_number": r.page_number,
                    "score": round(r.page_score, 3),
                }
                for r in results
            ],
        }


@app.get("/api/mayan/status")
async def mayan_status():
    """Check if Mayan EDMS is reachable."""
    mayan_url = os.environ.get("MAYAN_API_URL", "http://localhost:8000/api/v4")
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{mayan_url}/")
            return {"status": "connected", "url": mayan_url, "http_code": resp.status_code}
    except Exception as e:
        return {"status": "disconnected", "url": mayan_url, "error": str(e)}


# ============================================================
# AI Intelligence: Contextual Nudging & Semantic Batching
# ============================================================

@app.get("/api/nudge/{doc_id}")
async def nudge_similar_documents(doc_id: int, limit: int = 5):
    """Contextual Nudging: find documents similar to the one being viewed.

    Returns related documents with specific matching sections and pages.
    Uses document-level embeddings for fast doc-to-doc comparison,
    then drills into chunk-level matches for section citations.
    """
    document_key = make_document_key(str(doc_id), doc_id)

    # Get the document's aggregated embedding
    child_embeddings = search_engine.store.get_child_embeddings(document_key)
    if not child_embeddings:
        return {"doc_id": doc_id, "related": [], "message": "Document not indexed yet"}

    import numpy as np
    doc_embedding = np.mean(child_embeddings, axis=0).tolist()

    # Find similar documents (fast: doc-level comparison)
    similar_docs = search_engine.store.find_similar_documents(
        query_embedding=doc_embedding,
        exclude_document_key=document_key,
        limit=limit,
    )

    # For each similar document, find the specific matching sections
    results = []
    for doc in similar_docs:
        if doc["similarity"] < 0.3:
            continue

        # Find chunk-level matches between source doc and this related doc
        matching_sections = search_engine.store.find_similar_chunks_cross_doc(
            query_embedding=doc_embedding,
            exclude_document_key=document_key,
            limit=5,
        )

        # Filter to just this document's chunks
        doc_matches = [
            {
                "section": m["section"],
                "page": m["page_number"],
                "similarity": round(float(m["similarity"]), 3),
                "preview": m["content"][:200],
            }
            for m in matching_sections
            if m["document_key"] == doc["document_key"]
        ][:3]

        results.append({
            "doc_id": doc["mayan_doc_id"],
            "doc_name": doc["document_name"],
            "doc_type": doc["doc_type"],
            "similarity": round(float(doc["similarity"]), 3),
            "matching_sections": doc_matches,
        })

    return {"doc_id": doc_id, "related": results}


@app.post("/api/batch/summarize")
async def batch_summarize(body: dict):
    """Semantic Batching: generate an executive briefing for a collection of documents.

    Accepts either a cabinet name or a list of document IDs.
    Pulls parent chunks (high-density info) and synthesizes via Groq.
    """
    cabinet = body.get("cabinet")
    doc_ids = body.get("document_ids", [])
    prompt_hint = body.get("prompt", "")

    # Resolve document keys
    if cabinet:
        # Get documents in this cabinet from Mayan
        try:
            token = await mayan._get_token()
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(
                    f"{mayan.base_url}/cabinets/?format=json",
                    headers=mayan._auth_headers(),
                )
                cabinets = resp.json().get("results", [])
                cab_id = None
                for c in cabinets:
                    if c["label"].lower() == cabinet.lower():
                        cab_id = c["id"]
                        break
                if cab_id:
                    resp = await client.get(
                        f"{mayan.base_url}/cabinets/{cab_id}/documents/?format=json",
                        headers=mayan._auth_headers(),
                    )
                    doc_ids = [d["id"] for d in resp.json().get("results", [])]
        except Exception as e:
            return {"error": f"Failed to fetch cabinet: {e}"}

    if not doc_ids:
        return {"error": "No documents specified. Provide 'cabinet' or 'document_ids'."}

    # Get document keys for these Mayan IDs
    document_keys = [make_document_key(str(did), did) for did in doc_ids]

    # Pull parent chunks only — high-density information packets
    parents = search_engine.store.get_parent_chunks_for_documents(document_keys)
    if not parents:
        return {"error": "No indexed content found for these documents."}

    # Group by document for structured context
    from collections import defaultdict
    by_doc = defaultdict(list)
    for p in parents:
        by_doc[p["document_name"]].append(p)

    # Build context for LLM (parent chunks grouped by document)
    context_parts = []
    total_chars = 0
    for doc_name, doc_parents in by_doc.items():
        doc_section = f"=== {doc_name} ===\n"
        for p in doc_parents:
            section_text = f"[{p['section']}, Page {p['page_number']}]: {p['content'][:500]}\n"
            if total_chars + len(section_text) > 12000:
                break
            doc_section += section_text
            total_chars += len(section_text)
        context_parts.append(doc_section)
        if total_chars > 12000:
            break

    context = "\n".join(context_parts)

    # Generate briefing via Groq
    default_prompt = (
        f"Analyze these {len(by_doc)} documents. Provide an executive briefing that includes: "
        "1) A 3-4 sentence overall summary. "
        "2) Key themes or topics across the documents. "
        "3) Important entities (people, organizations, dates, amounts). "
        "4) Notable differences or contradictions between documents. "
        "Cite specific documents and pages for each finding."
    )

    groq_prompt = prompt_hint if prompt_hint else default_prompt

    api_key = os.environ.get("GROQ_API_KEY")
    briefing = None
    if api_key:
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json={
                        "model": "llama-3.3-70b-versatile",
                        "messages": [
                            {"role": "system", "content": "You are a document analyst. Provide structured executive briefings based on document excerpts. Always cite document names and page numbers."},
                            {"role": "user", "content": f"{groq_prompt}\n\nDOCUMENTS:\n{context}"},
                        ],
                        "max_tokens": 1000,
                        "temperature": 0,
                    },
                )
                if resp.status_code == 200:
                    briefing = resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            briefing = f"LLM error: {e}"

    return {
        "documents_analyzed": len(by_doc),
        "total_sections": len(parents),
        "document_names": list(by_doc.keys()),
        "briefing": briefing or "Groq API key not set. Cannot generate briefing.",
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 1234))
    uvicorn.run(app, host="0.0.0.0", port=port)
