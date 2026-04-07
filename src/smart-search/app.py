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

load_dotenv(override=False)  # Don't override Docker Compose env vars

from pdf_processor import PDFProcessor
from search_engine import SearchEngine
from chunking_pipeline import ChunkingPipeline
from mayan_bridge import MayanBridge
from storage_pg import make_document_key
from excel_parser import ExcelParser, is_excel_file, smart_detect
from excel_enricher import ExcelEnricher
from excel_storage import ExcelStorageManager, classify_query
from pipeline_debug import PipelineDebugger
from excel_agent import ExcelSQLAgent
from excel_routes import router as excel_router
from pdf_table_extractor import PDFTableExtractor

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
pdf_table_extractor = PDFTableExtractor()
excel_inspection_cache: dict[str, dict] = {}

from keyword_extractor import KeywordExtractor
from metadata_router import MetadataRouter

keyword_extractor = KeywordExtractor()
metadata_router = MetadataRouter()

# Pre-load embedding model at startup (avoids OOM during first sync)
search_engine._load_model()


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
    """Upload a PDF or Excel file and index it for searching.

    Smart detection: checks actual file content, not just the extension.
    A PDF named 'data.xlsx' will be routed to the PDF pipeline.
    """
    file_path = UPLOAD_DIR / file.filename
    content = await file.read()
    file_path.write_bytes(content)

    # Detect actual file type from content, not extension
    actual_type = smart_detect(str(file_path))
    print(f"[Upload] {file.filename}: extension={Path(file.filename).suffix}, detected={actual_type}")

    if actual_type == "excel":
        return _index_excel(str(file_path), file.filename)
    elif actual_type == "pdf":
        return _index_pdf(str(file_path), file.filename)
    else:
        raise HTTPException(400, f"Unsupported file type. Detected: {actual_type}. Supported: PDF, Excel (.xlsx, .xls, .csv)")


def _extract_raw_text_pdf(structure) -> str:
    """Get clean raw text from a DocumentStructure (before chunking/tagging)."""
    parts = []
    for page in structure.pages:
        for section in page.sections:
            if section.content:
                parts.append(section.content)
    return "\n".join(parts)


def _extract_raw_text_excel(dataframes: dict) -> str:
    """Get clean raw text from DataFrames (before semantic row tagging)."""
    parts = []
    for sheet_name, df in dataframes.items():
        parts.append(f"Sheet: {sheet_name}")
        for col in df.columns:
            if col == "__section__":
                continue
            # Add column name
            parts.append(col)
            # Add unique values from text columns
            if df[col].dtype == "object":
                for val in df[col].dropna().unique()[:30]:
                    parts.append(str(val))
            # Add numeric values as strings
            elif df[col].dtype.kind in ("i", "f"):
                for val in df[col].dropna().unique()[:20]:
                    parts.append(str(val))
    return "\n".join(parts)


def _build_and_store_route(
    document_key: str,
    document_id: int | None,
    filename: str,
    raw_text: str,
    cabinet_landmark: dict | None,
    chunk_count: int = 0,
) -> dict:
    """Extract keywords from raw text and store the document route.

    This runs BEFORE chunking — uses clean document text, not tagged chunks.
    """
    # Extract keywords from raw document text (clean, no pipeline tags)
    keywords = keyword_extractor.extract(raw_text)
    print(f"[Route] Extracted {len(keywords)} keywords from {filename}")

    # Build summary text for embedding
    company = cabinet_landmark.get("company_name", "") if cabinet_landmark else ""
    cabinet_path = cabinet_landmark.get("cabinet_path", "") if cabinet_landmark else ""
    summary_parts = [
        f"Company: {company}",
        f"Document: {filename}",
        f"Location: {cabinet_path}",
        f"Keywords: {' '.join(keywords[:80])}",
    ]
    summary_text = "\n".join(summary_parts)

    # Embed the summary
    search_engine._load_model()
    summary_emb = list(search_engine.model.embed([summary_text]))[0].tolist()

    # Store the route
    metadata_router.store_route(
        document_key=document_key,
        document_id=document_id,
        document_name=filename,
        keywords=keywords,
        summary_text=summary_text,
        summary_embedding=summary_emb,
        cabinet_landmark=cabinet_landmark,
        chunk_count=chunk_count,
    )

    return {"keywords_count": len(keywords), "keywords_sample": keywords[:10]}


def _cache_excel_inspection(
    document_key: str,
    stage1_result,
    classification_meta: dict,
    semantic_rows: list[dict],
    schema_description: str,
) -> None:
    """Cache Stage 1/2 artifacts for the Excel inspection UI."""
    stage1_payload = None
    if stage1_result is not None and hasattr(stage1_result, "to_dict"):
        stage1_payload = stage1_result.to_dict(
            max_cells_per_sheet=500,
            max_rows_per_table=200,
        )

    excel_inspection_cache[document_key] = {
        "document_key": document_key,
        "stage1": stage1_payload,
        "classification": classification_meta,
        "stage2": {
            "total_semantic_rows": len(semantic_rows),
            "semantic_rows": semantic_rows[:500],
        },
        "schema": schema_description,
    }


def _stamp_cabinet_on_chunks(doc_id: int, cabinet_landmark: dict, filename: str = "") -> None:
    """Stamp full route on all chunks for a document in pgvector.

    Every chunk gets a complete route that includes:
      - cabinet_ids:    [3, 8, 12, 13, 15, 16]    ← all cabinet ancestors
      - document_id:    14                          ← the document itself
      - document_key:   "mayan:14"                  ← pgvector document key
      - full_path_ids:  [3, 8, 12, 13, 15, 16, 14] ← cabinets + doc in one array
      - full_path:      "Apple Inc. / Q2 2025 / ... / Recipegenie.pdf"
      - cabinet_depth:  [{id, label, level}, ...]   ← human-readable hierarchy
      - company_id:     3                           ← root company
      - company_name:   "Apple Inc."
      - cabinet_id:     16                          ← leaf cabinet

    Query at any level:
      WHERE metadata->'cabinet_ids' @> '[8]'        → all Q2 2025 docs
      WHERE metadata->'full_path_ids' @> '[14]'     → this specific document
      WHERE metadata->>'company_id' = '3'           → all Apple docs
    """
    document_key = make_document_key(str(doc_id), doc_id)
    try:
        if not search_engine.store._pg_available:
            return
        import json as _json

        hierarchy = cabinet_landmark.get("hierarchy", [])
        cabinet_ids = [node["id"] for node in hierarchy]
        full_path_ids = cabinet_ids + [doc_id]

        # Build full path string including document name
        cabinet_path = cabinet_landmark.get("cabinet_path", "")
        doc_name = filename or f"doc:{doc_id}"
        full_path = f"{cabinet_path} / {doc_name}" if cabinet_path else doc_name

        stamp = {
            "cabinet_id": cabinet_landmark["cabinet_id"],
            "cabinet_path": cabinet_path,
            "company_id": cabinet_landmark["company_id"],
            "company_name": cabinet_landmark["company_name"],
            "cabinet_ids": cabinet_ids,
            "cabinet_depth": hierarchy,
            "document_id": doc_id,
            "document_key": document_key,
            "full_path_ids": full_path_ids,
            "full_path": full_path,
        }

        with search_engine.store._connect() as conn:
            conn.execute(
                f"""
                UPDATE {search_engine.store.schema}.chunks
                SET metadata = metadata || %s::jsonb
                WHERE document_key = %s
                """,
                (_json.dumps(stamp), document_key),
            )
        depth_str = " → ".join(f"{n['label']}(id={n['id']})" for n in hierarchy)
        print(f"[Route] Stamped [{depth_str} → {doc_name}(doc={doc_id})] on all chunks for {document_key}")
    except Exception as e:
        print(f"[Route] Stamp failed (non-fatal): {e}")


def _index_pdf(pdf_path: str, filename: str, mayan_doc_id: int | None = None) -> dict:
    """Index a PDF file through the existing pipeline.

    Also detects tables in PDF content and extracts them into SQLite
    so the SQL agent can query tabular data even from PDFs.
    """
    pages = pdf_processor.extract_pages(pdf_path)
    page_count = pdf_processor.get_page_count(pdf_path)

    structure = extract_structure(pdf_path)
    doc_type = structure.doc_type
    document_key = make_document_key(filename, mayan_doc_id)

    # Stage 1.5: Extract keywords from RAW text (before chunking/tagging)
    raw_text = _extract_raw_text_pdf(structure)

    chunks = chunking_pipeline.chunk_document(structure, filename, document_key=document_key)
    search_engine.index(chunks, pages, document_id=mayan_doc_id, document_name=filename)

    # Build and store route (keywords + embedding + cabinet landmark)
    route_info = _build_and_store_route(
        document_key, mayan_doc_id, filename, raw_text, None, len(chunks),
    )

    current_pdf["path"] = pdf_path
    current_pdf["name"] = filename
    current_pdf["page_count"] = page_count

    # --- Extract tables from PDF into SQLite for SQL queries ---
    has_sql = False
    table_count = 0
    try:
        dataframes = pdf_table_extractor.extract_tables(chunks)
        if dataframes:
            # Classify columns + generate schema (reuse Excel enricher)
            excel_enricher.classify_and_summarize(dataframes)
            excel_storage.store_excel(document_key, filename, dataframes)
            has_sql = True
            table_count = len(dataframes)
            print(f"[PDF→SQL] Extracted {table_count} tables from {filename} into SQLite")
    except Exception as e:
        print(f"[PDF→SQL] Table extraction failed (non-fatal): {e}")

    return {
        "filename": filename,
        "page_count": page_count,
        "indexed_pages": len(pages),
        "total_lines": sum(len(p.lines) for p in pages),
        "doc_type": doc_type,
        "total_chunks": len(chunks),
        "sql_queryable": has_sql,
        "tables_extracted": table_count,
    }


def _index_excel(file_path: str, filename: str, mayan_doc_id: int | None = None) -> dict:
    """Index an Excel file: parse → classify → semantic rows → embed → store.

    Option B pipeline: semantic rows ARE the chunks. No Gemini extraction needed.
    Each row becomes a child chunk with role tags; each sheet gets a parent chunk
    with summary context.
    """
    debug = PipelineDebugger(filename)

    # Stage 1: Adaptive parsing (openpyxl/pandas — no API calls)
    dataframes, formulas, cell_dna = excel_parser.parse(file_path)
    excel_enricher.set_stage1_result(excel_parser.last_stage1_result)
    debug.dump_stage1(dataframes, formulas)

    # Stage 1.5a: Extract keywords from RAW DataFrames (before tagging)
    raw_text = _extract_raw_text_excel(dataframes)

    # Stage 1.5b: LLM column classification + sheet summary (DNA-accelerated)
    classification_meta = excel_enricher.classify_and_summarize(dataframes, cell_dna)
    debug.dump_stage1_5(classification_meta)

    # Stage 2: Generate schema description (for SQL agent)
    schema_description = excel_enricher.generate_all_schemas(dataframes)

    # Stage 2 debug: dump semantic rows for inspection
    all_semantic_rows = []
    for sheet_name, df in dataframes.items():
        sheet_formulas = formulas.get(sheet_name, [])
        rows = excel_enricher.generate_semantic_rows(df, sheet_name, sheet_formulas)
        all_semantic_rows.extend(rows)
    debug.dump_stage2(all_semantic_rows, schema_description)
    _cache_excel_inspection(
        document_key=make_document_key(filename, mayan_doc_id),
        stage1_result=excel_parser.last_stage1_result,
        classification_meta=classification_meta,
        semantic_rows=all_semantic_rows,
        schema_description=schema_description,
    )

    # Stage 3: Build chunks directly from semantic rows (NO Gemini, NO extract_structure)
    document_key = make_document_key(filename, mayan_doc_id)
    chunks = excel_enricher.build_chunks(
        dataframes, formulas, filename, document_key,
        schema_description, mayan_doc_id,
    )

    debug.dump_stage3(chunks)
    debug.print_summary()

    # Stage 4: Embed and store in pgvector
    search_engine.index(chunks, pages=None, document_id=mayan_doc_id, document_name=filename)

    # Stage 4.5: Build and store route (keywords from RAW text + cabinet landmark)
    route_info = _build_and_store_route(
        document_key, mayan_doc_id, filename, raw_text, None, len(chunks),
    )

    # Stage 5: Store DataFrames in SQLite for SQL queries
    has_sql = False
    if dataframes:
        try:
            excel_storage.store_excel(
                document_key,
                filename,
                dataframes,
                semantic_rows=all_semantic_rows,
            )
            has_sql = True
        except Exception as e:
            print(f"[Excel] SQLite storage failed (non-fatal): {e}")

    parent_count = sum(1 for c in chunks if c.metadata.get("chunk_type") == "parent")
    child_count = sum(1 for c in chunks if c.metadata.get("chunk_type") == "child")

    return {
        "filename": filename,
        "document_key": document_key,
        "file_type": "spreadsheet",
        "sheets": {name: {"rows": len(df), "columns": len(df.columns)} for name, df in dataframes.items()},
        "total_rows": sum(len(df) for df in dataframes.values()),
        "total_semantic_rows": len(all_semantic_rows),
        "total_chunks": len(chunks),
        "parent_chunks": parent_count,
        "child_chunks": child_count,
        "doc_type": "spreadsheet",
        "schema_description": schema_description,
        "classification": classification_meta,
        "sql_queryable": has_sql,
        "debug_dir": str(debug.output_dir) if debug.output_dir else None,
    }


@app.get("/api/cabinets")
async def list_cabinets():
    """List all cabinets as a tree for the UI dropdown."""
    try:
        cabinets = await mayan.list_cabinets()
        # Build tree structure
        tree = []
        top_level = [c for c in cabinets if c.get("parent_id") is None]
        children_map: dict[int, list] = {}
        for c in cabinets:
            pid = c.get("parent_id")
            if pid:
                children_map.setdefault(pid, []).append(c)

        for company in top_level:
            entry = {
                "id": company["id"],
                "label": company["label"],
                "level": "company",
                "children": [],
            }
            for child in children_map.get(company["id"], []):
                entry["children"].append({
                    "id": child["id"],
                    "label": child["label"],
                    "level": "contract",
                })
            tree.append(entry)

        return {"cabinets": tree}
    except Exception as e:
        return {"cabinets": [], "error": str(e)}


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
    """Two-stage search: find the right document first, then search inside it.

    Stage 1: MetadataRouter (keyword + vector + RRF on document_routes)
    Stage 2: Scoped chunk search (only within winning documents)
    """
    query = body.get("query", "").strip()
    if not query:
        raise HTTPException(400, "Query is required")
    if not current_pdf["path"] and not body.get("document_id") and not search_engine.has_documents():
        raise HTTPException(400, "No document uploaded")

    document_id = body.get("document_id")
    cabinet_id = body.get("cabinet_id")

    # ── STAGE 1: Find the right document(s) via metadata routing ──
    search_engine._load_model()
    query_embedding = list(search_engine.model.embed([query]))[0].tolist()

    routed_docs = metadata_router.route(
        query=query,
        query_embedding=query_embedding,
        top_k=5,
        cabinet_id=int(cabinet_id) if cabinet_id else None,
    )

    # Log routing results
    if routed_docs:
        print(f"[Route] Query: '{query[:60]}' → Stage 1 results:")
        for i, rd in enumerate(routed_docs):
            print(f"  #{i+1} {rd['document_name']:50s} score={rd['rrf_score']:.6f} path={rd.get('full_path','?')}")
    else:
        print(f"[Route] No routes matched for: '{query[:60]}' — falling back to unscoped search")

    # Build scoped document keys — ONLY the winner from Stage 1
    scoped_keys = None
    if routed_docs:
        scoped_keys = [routed_docs[0]["document_key"]]

    query_mode = classify_query(query)

    if routed_docs and query_mode == "provenance":
        target_key = routed_docs[0]["document_key"]
        if excel_storage.has_sql_database(target_key):
            result = excel_storage.answer_provenance(target_key, query)
            entry = excel_storage.get_registry_entry(target_key)
            doc_name = entry["document_name"] if entry else target_key
            return {
                "query": query,
                "topic": search_engine.extract_topic(query),
                "total_results": 1,
                "results": [],
                "ai_summary": {
                    "text": result.get("answer", "No answer"),
                    "model": "Provenance Store",
                    "sources": [{"document_name": doc_name, "page_number": 1}],
                },
                "provenance_trace": {
                    "results": result.get("results", []),
                },
            }

    # Check if winner has SQL database for computation queries
    if routed_docs and query_mode == "sql":
        target_key = routed_docs[0]["document_key"]
        if excel_storage.has_sql_database(target_key):
            schema = excel_storage.get_schema_description(target_key)
            result = excel_agent.ask(
                question=query,
                schema=schema,
                execute_fn=lambda sql: excel_storage.execute_sql(target_key, sql),
            )

            entry = excel_storage.get_registry_entry(target_key)
            doc_name = entry["document_name"] if entry else target_key

        if target_key and excel_storage.has_sql_database(target_key):
            schema = excel_storage.get_schema_description(target_key)
            result = excel_agent.ask(
                question=query,
                schema=schema,
                execute_fn=lambda sql: excel_storage.execute_sql(target_key, sql),
            )

            entry = excel_storage.get_registry_entry(target_key)
            doc_name = entry["document_name"] if entry else target_key
            answer_text = result.get("answer", "No answer")
            sql_used = result.get("sql", "")
            return {
                "query": query,
                "topic": search_engine.extract_topic(query),
                "total_results": 1,
                "results": [],
                "ai_summary": {
                    "text": answer_text,
                    "model": result.get("model", "SQL Agent"),
                    "sources": [{"document_name": doc_name, "page_number": 1}],
                },
                "sql_trace": {
                    "query_type": "sql",
                    "sql": sql_used,
                    "results": result.get("results", []),
                    "attempts": result.get("attempts", 1),
                    "document": doc_name,
                },
                "routing": {
                    "stage": "metadata_router",
                    "matched_docs": [
                        {"document": rd["document_name"], "score": rd["rrf_score"],
                         "path": rd.get("full_path"), "path_ids": rd.get("full_path_ids")}
                        for rd in routed_docs[:3]
                    ],
                },
            }

    # ── STAGE 2: Scoped chunk search (only within routed documents) ──
    results = search_engine.search(query, top_k=5, document_id=document_id, document_keys=scoped_keys)

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
        "routing": {
            "stage": "metadata_router",
            "matched_docs": [
                {"document": rd["document_name"], "score": rd["rrf_score"],
                 "path": rd.get("full_path"), "path_ids": rd.get("full_path_ids")}
                for rd in routed_docs[:3]
            ] if routed_docs else [],
            "scoped_to": scoped_keys,
        },
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

        # 1b. Get cabinet landmark — stamps every chunk with its location
        cabinet_landmark = await mayan.get_document_cabinet_path(doc_id)
        if cabinet_landmark:
            print(f"[Cabinet] {mayan_doc.label} → {cabinet_landmark['cabinet_path']}")
        else:
            print(f"[Cabinet] {mayan_doc.label} → no cabinet assigned")

        # Route based on actual content, not extension
        actual_type = smart_detect(mayan_doc.file_path)
        print(f"[MayanSync] {mayan_doc.label}: extension={Path(mayan_doc.file_path).suffix}, detected={actual_type}")

        if actual_type == "excel":
            result = _index_excel(mayan_doc.file_path, mayan_doc.label, mayan_doc_id=doc_id)
            # Stamp cabinet landmark on chunks + update route
            if cabinet_landmark:
                _stamp_cabinet_on_chunks(doc_id, cabinet_landmark, mayan_doc.label)
                metadata_router.update_cabinet(f"mayan:{doc_id}", cabinet_landmark)
            await mayan.classify_document(doc_id=doc_id, rag_doc_type="spreadsheet", tags=["AI: Spreadsheet"])
            result["mayan_doc_id"] = doc_id
            result["mayan_metadata"] = mayan_doc.metadata
            result["cabinet"] = cabinet_landmark
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

        # 4a. Extract keywords from RAW text (before chunking)
        raw_text = _extract_raw_text_pdf(structure)
        route_info = _build_and_store_route(
            document_key, doc_id, mayan_doc.label, raw_text, cabinet_landmark,
            chunk_count=0,  # updated after chunking
        )

        chunks = chunking_pipeline.chunk_document(structure, mayan_doc.label, document_key=document_key)

        # Enrich chunks with Mayan metadata
        for chunk in chunks:
            chunk.metadata["mayan_doc_id"] = doc_id
            chunk.metadata["mayan_tags"] = mayan_doc.metadata.get("tags", [])
            chunk.metadata["mayan_cabinet"] = mayan_doc.metadata.get("cabinets", [])
            chunk.metadata["text_source_mode"] = source_profile.get("mode")

        # 5. Index
        search_engine.index(chunks, pages, document_id=doc_id, document_name=mayan_doc.label)

        # 5a. Stamp cabinet landmark on chunks + update route
        if cabinet_landmark:
            _stamp_cabinet_on_chunks(doc_id, cabinet_landmark, mayan_doc.label)
            metadata_router.update_cabinet(document_key, cabinet_landmark)

        # 5b. Extract tables from PDF into SQLite for SQL queries
        try:
            dataframes = pdf_table_extractor.extract_tables(chunks)
            if dataframes:
                excel_enricher.classify_and_summarize(dataframes)
                excel_storage.store_excel(document_key, mayan_doc.label, dataframes)
                print(f"[PDF→SQL] Extracted {len(dataframes)} tables from {mayan_doc.label} into SQLite")
        except Exception as e:
            print(f"[PDF→SQL] Table extraction failed (non-fatal): {e}")

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
        cabinet_landmark = await mayan.get_document_cabinet_path(doc_id)
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

        # Route based on actual content, not extension
        actual_type = smart_detect(mayan_doc.file_path)
        print(f"[Webhook] {mayan_doc.label}: extension={Path(mayan_doc.file_path).suffix}, detected={actual_type}")

        if actual_type == "excel":
            result = _index_excel(mayan_doc.file_path, mayan_doc.label, mayan_doc_id=doc_id)
            if cabinet_landmark:
                _stamp_cabinet_on_chunks(doc_id, cabinet_landmark, mayan_doc.label)
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

        # Extract keywords from RAW text before chunking
        raw_text = _extract_raw_text_pdf(structure)
        _build_and_store_route(
            document_key, doc_id, mayan_doc.label, raw_text, cabinet_landmark,
        )

        chunks = chunking_pipeline.chunk_document(structure, mayan_doc.label, document_key=document_key)

        for chunk in chunks:
            chunk.metadata["mayan_doc_id"] = doc_id
            chunk.metadata["mayan_version_id"] = mayan_doc.version_id
            chunk.metadata["text_source_mode"] = source_profile.get("mode")

        search_engine.index(chunks, pages, document_id=doc_id, document_name=mayan_doc.label)

        # Stamp cabinet landmark on chunks
        if cabinet_landmark:
            _stamp_cabinet_on_chunks(doc_id, cabinet_landmark, mayan_doc.label)
            metadata_router.update_cabinet(document_key, cabinet_landmark)

        # Extract tables from PDF into SQLite for SQL queries
        try:
            dataframes = pdf_table_extractor.extract_tables(chunks)
            if dataframes:
                excel_enricher.classify_and_summarize(dataframes)
                excel_storage.store_excel(document_key, mayan_doc.label, dataframes)
                print(f"[PDF→SQL] Extracted {len(dataframes)} tables from {mayan_doc.label} into SQLite")
        except Exception as e:
            print(f"[PDF→SQL] Table extraction failed (non-fatal): {e}")

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


@app.get("/api/excel/inspect/{document_key}")
async def excel_inspect(document_key: str):
    """Return cached Stage 1/2 artifacts for the Excel inspection UI."""
    payload = excel_inspection_cache.get(document_key)
    if not payload:
        raise HTTPException(404, "No inspection data found for this document.")
    return payload


@app.get("/rag-debug", response_class=HTMLResponse)
async def rag_debug_ui():
    """RAG Pipeline Debug UI — shows Stage 1, 2, 3 output."""
    return Path("static/rag_debug.html").read_text()


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

    if query_type == "provenance":
        result = excel_storage.answer_provenance(document_key, question)
        trace["results"] = result.get("results", [])
        return {
            "query_type": "provenance",
            "question": question,
            "answer": result.get("answer", "No answer"),
            "trace": trace,
        }
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

    if query_type == "provenance":
        result = excel_storage.answer_provenance(document_key, question)
        return {
            "query_type": "provenance",
            "question": question,
            "document_key": document_key,
            **result,
        }
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
