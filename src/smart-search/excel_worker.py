"""Celery worker for background Excel indexing.

Runs the full pipeline (parse -> enrich -> chunk -> embed -> store) as an
async task, reporting progress at each stage.

Docker Compose Addition
-----------------------
Add this service to docker/docker-compose.yml alongside rag-sidecar:

  excel-worker:
    build:
      context: ../src/smart-search
      dockerfile: Dockerfile
    container_name: excel-worker
    command: ["celery", "-A", "celery_config", "worker", "--loglevel=info", "--concurrency=2", "-Q", "excel"]
    deploy:
      resources:
        limits:
          memory: 4G
    environment:
      CELERY_BROKER_URL: amqp://mayan:mayan@rabbitmq:5672/mayan
      CELERY_RESULT_BACKEND: redis://redis:6379/2
      GROQ_API_KEY: ${GROQ_API_KEY:-}
      RAG_PGHOST: postgresql
      RAG_PGPORT: 5432
      RAG_PGDATABASE: mayan
      RAG_PGUSER: mayan
      RAG_PGPASSWORD: mayan
      RAG_PG_SCHEMA: rag
    volumes:
      - rag-uploads:/app/uploads
      - rag-cache:/app/cache
    depends_on:
      rabbitmq:
        condition: service_healthy
      redis:
        condition: service_healthy
      postgresql:
        condition: service_healthy
    restart: unless-stopped
"""

from __future__ import annotations

import logging
import traceback
from pathlib import Path

from celery_config import app

logger = logging.getLogger(__name__)


def _build_pipeline():
    """Lazily import and construct pipeline components.

    Heavy imports (pandas, openpyxl, fastembed, psycopg) happen once per
    worker process, not at module import time.  This keeps the Celery
    beat/flower processes lightweight.
    """
    from excel_parser import ExcelParser
    from excel_enricher import ExcelEnricher
    from excel_storage import ExcelStorageManager
    from search_engine import SearchEngine
    from storage_pg import make_document_key

    return {
        "excel_parser": ExcelParser(),
        "excel_enricher": ExcelEnricher(),
        "excel_storage": ExcelStorageManager(),
        "search_engine": SearchEngine(),
        "make_document_key": make_document_key,
    }


# Module-level cache so we build once per worker process.
_pipeline: dict | None = None


def _get_pipeline() -> dict:
    global _pipeline
    if _pipeline is None:
        _pipeline = _build_pipeline()
    return _pipeline


# ------------------------------------------------------------------
# Celery task
# ------------------------------------------------------------------


@app.task(bind=True, name="excel.index", queue="excel", max_retries=3)
def index_excel_task(
    self,
    file_path: str,
    filename: str,
    mayan_doc_id: int | None = None,
) -> dict:
    """Background task: parse -> enrich -> chunk -> embed -> store.

    Progress is reported at each stage via ``self.update_state`` so callers
    can poll ``/api/excel/status/{task_id}`` for real-time feedback.

    Parameters
    ----------
    file_path:
        Absolute path to the uploaded Excel file on the shared volume.
    filename:
        Original filename (used as document name in the index).
    mayan_doc_id:
        Optional Mayan EDMS document ID for cross-referencing.

    Returns
    -------
    dict with keys: status, filename, file_type, sheets, total_rows,
    total_semantic_rows, total_chunks, doc_type, schema_description.
    """
    pipeline = _get_pipeline()
    excel_parser = pipeline["excel_parser"]
    excel_enricher = pipeline["excel_enricher"]
    excel_storage = pipeline["excel_storage"]
    search_engine = pipeline["search_engine"]
    make_document_key = pipeline["make_document_key"]

    try:
        # ----------------------------------------------------------
        # Stage 1: Adaptive parsing (20%)
        # ----------------------------------------------------------
        self.update_state(
            state="PROGRESS",
            meta={"stage": "parsing", "progress": 10, "message": f"Parsing {filename}..."},
        )
        logger.info("Stage 1: Parsing %s", filename)

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        from pipeline_debug import PipelineDebugger
        debug = PipelineDebugger(filename)

        dataframes, formulas, cell_dna = excel_parser.parse(file_path)
        excel_enricher.set_stage1_result(excel_parser.last_stage1_result)
        debug.dump_stage1(dataframes, formulas)

        self.update_state(
            state="PROGRESS",
            meta={"stage": "parsing", "progress": 20, "message": f"Parsed {len(dataframes)} sheet(s)."},
        )

        # ----------------------------------------------------------
        # Stage 1.5: Column classification + sheet summary (30%)
        # ----------------------------------------------------------
        self.update_state(
            state="PROGRESS",
            meta={"stage": "classifying", "progress": 25, "message": "Classifying columns..."},
        )
        logger.info("Stage 1.5: Classifying columns in %s", filename)

        classification_meta = excel_enricher.classify_and_summarize(dataframes, cell_dna)
        debug.dump_stage1_5(classification_meta)

        # ----------------------------------------------------------
        # Stage 2: Enrichment + schema (45%)
        # ----------------------------------------------------------
        self.update_state(
            state="PROGRESS",
            meta={"stage": "enriching", "progress": 35, "message": "Generating semantic rows..."},
        )
        logger.info("Stage 2: Enriching %s", filename)

        schema_description = excel_enricher.generate_all_schemas(dataframes)

        # Dump semantic rows for debug inspection
        all_semantic_rows: list[dict] = []
        for sheet_name, df in dataframes.items():
            sheet_formulas = formulas.get(sheet_name, [])
            rows = excel_enricher.generate_semantic_rows(df, sheet_name, sheet_formulas)
            all_semantic_rows.extend(rows)
        debug.dump_stage2(all_semantic_rows, schema_description)

        self.update_state(
            state="PROGRESS",
            meta={"stage": "enriching", "progress": 45, "message": f"{len(all_semantic_rows)} semantic rows."},
        )

        # ----------------------------------------------------------
        # Stage 3: Build chunks from semantic rows (60%)
        # ----------------------------------------------------------
        self.update_state(
            state="PROGRESS",
            meta={"stage": "chunking", "progress": 50, "message": "Building chunks from semantic rows..."},
        )
        logger.info("Stage 3: Building chunks for %s", filename)

        document_key = make_document_key(filename, mayan_doc_id)
        chunks = excel_enricher.build_chunks(
            dataframes, formulas, filename, document_key,
            schema_description, mayan_doc_id,
        )

        debug.dump_stage3(chunks)
        debug.print_summary()

        self.update_state(
            state="PROGRESS",
            meta={"stage": "chunking", "progress": 60, "message": f"Created {len(chunks)} chunks."},
        )

        # ----------------------------------------------------------
        # Stage 4: Embedding + storing (100%)
        # ----------------------------------------------------------
        self.update_state(
            state="PROGRESS",
            meta={"stage": "embedding", "progress": 70, "message": "Generating embeddings..."},
        )
        logger.info("Stage 4: Embedding %s (%d chunks)", filename, len(chunks))

        self.update_state(
            state="PROGRESS",
            meta={"stage": "storing", "progress": 85, "message": "Writing to PostgreSQL + pgvector..."},
        )
        logger.info("Stage 5: Storing %s", filename)

        search_engine.index(
            chunks, pages=None,
            document_id=mayan_doc_id, document_name=filename,
        )

        if dataframes:
            excel_storage.store_excel(
                document_key,
                filename,
                dataframes,
                semantic_rows=all_semantic_rows,
            )

        parent_count = sum(1 for c in chunks if c.metadata.get("chunk_type") == "parent")
        child_count = sum(1 for c in chunks if c.metadata.get("chunk_type") == "child")

        logger.info(
            "Completed indexing %s: %d parent + %d child chunks",
            filename, parent_count, child_count,
        )

        return {
            "status": "completed",
            "filename": filename,
            "file_type": "spreadsheet",
            "sheets": {
                name: {"rows": len(df), "columns": len(df.columns)}
                for name, df in dataframes.items()
            },
            "total_rows": sum(len(df) for df in dataframes.values()),
            "total_semantic_rows": len(all_semantic_rows),
            "total_chunks": len(chunks),
            "parent_chunks": parent_count,
            "child_chunks": child_count,
            "doc_type": "spreadsheet",
            "schema_description": schema_description,
            "classification": classification_meta,
        }

    except FileNotFoundError:
        logger.error("File not found: %s", file_path)
        raise

    except Exception as exc:
        logger.exception("Failed to index %s", filename)
        self.update_state(
            state="FAILURE",
            meta={
                "stage": "error",
                "progress": 0,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        # Retry with exponential backoff: 60s, 180s, 540s
        raise self.retry(exc=exc, countdown=60 * (3 ** self.request.retries))
