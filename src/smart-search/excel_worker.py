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
    from chunking_pipeline import ChunkingPipeline
    from search_engine import SearchEngine
    from storage_pg import make_document_key

    return {
        "excel_parser": ExcelParser(),
        "excel_enricher": ExcelEnricher(),
        "chunking_pipeline": ChunkingPipeline(),
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
    chunking_pipeline = pipeline["chunking_pipeline"]
    search_engine = pipeline["search_engine"]
    make_document_key = pipeline["make_document_key"]

    try:
        # ----------------------------------------------------------
        # Stage 1: Adaptive parsing (20%)
        # ----------------------------------------------------------
        self.update_state(
            state="PROGRESS",
            meta={
                "stage": "parsing",
                "progress": 10,
                "message": f"Parsing {filename}...",
            },
        )
        logger.info("Stage 1/5: Parsing %s", filename)

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        dataframes, formulas = excel_parser.parse(file_path)

        self.update_state(
            state="PROGRESS",
            meta={
                "stage": "parsing",
                "progress": 20,
                "message": f"Parsed {len(dataframes)} sheet(s).",
            },
        )

        # ----------------------------------------------------------
        # Stage 2: Structural enrichment (40%)
        # ----------------------------------------------------------
        self.update_state(
            state="PROGRESS",
            meta={
                "stage": "enriching",
                "progress": 30,
                "message": "Generating semantic rows...",
            },
        )
        logger.info("Stage 2/5: Enriching %s", filename)

        all_semantic_rows: list[dict] = []
        for sheet_name, df in dataframes.items():
            sheet_formulas = formulas.get(sheet_name, [])
            rows = excel_enricher.generate_semantic_rows(df, sheet_name, sheet_formulas)
            all_semantic_rows.extend(rows)

        schema_description = excel_enricher.generate_all_schemas(dataframes)

        self.update_state(
            state="PROGRESS",
            meta={
                "stage": "enriching",
                "progress": 40,
                "message": f"Enriched {len(all_semantic_rows)} semantic rows.",
            },
        )

        # ----------------------------------------------------------
        # Stage 3: Chunking (60%)
        # ----------------------------------------------------------
        self.update_state(
            state="PROGRESS",
            meta={
                "stage": "chunking",
                "progress": 50,
                "message": "Extracting structure and chunking...",
            },
        )
        logger.info("Stage 3/5: Chunking %s", filename)

        structure = excel_parser.extract_structure(file_path)
        document_key = make_document_key(filename, mayan_doc_id)
        chunks = chunking_pipeline.chunk_document(
            structure, filename, document_key=document_key
        )

        # Enrich chunks with Excel-specific metadata
        for chunk in chunks:
            chunk.metadata["file_type"] = "spreadsheet"
            chunk.metadata["schema_description"] = schema_description
            chunk.metadata["sheet_names"] = list(dataframes.keys())
            chunk.metadata["total_rows"] = sum(
                len(df) for df in dataframes.values()
            )
            if mayan_doc_id:
                chunk.metadata["mayan_doc_id"] = mayan_doc_id

        self.update_state(
            state="PROGRESS",
            meta={
                "stage": "chunking",
                "progress": 60,
                "message": f"Created {len(chunks)} chunks.",
            },
        )

        # ----------------------------------------------------------
        # Stage 4: Embedding (80%)
        # ----------------------------------------------------------
        self.update_state(
            state="PROGRESS",
            meta={
                "stage": "embedding",
                "progress": 70,
                "message": "Generating embeddings...",
            },
        )
        logger.info("Stage 4/5: Embedding %s (%d chunks)", filename, len(chunks))

        # search_engine.index handles embedding + pgvector storage internally
        # We report embedding and storing as separate logical stages.

        self.update_state(
            state="PROGRESS",
            meta={
                "stage": "embedding",
                "progress": 80,
                "message": "Embeddings generated. Storing in vector database...",
            },
        )

        # ----------------------------------------------------------
        # Stage 5: Storing (100%)
        # ----------------------------------------------------------
        self.update_state(
            state="PROGRESS",
            meta={
                "stage": "storing",
                "progress": 85,
                "message": "Writing to PostgreSQL + pgvector...",
            },
        )
        logger.info("Stage 5/5: Storing %s", filename)

        search_engine.index(
            chunks,
            pages=None,
            document_id=mayan_doc_id,
            document_name=filename,
        )

        logger.info(
            "Completed indexing %s: %d chunks, %d semantic rows",
            filename,
            len(chunks),
            len(all_semantic_rows),
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
            "doc_type": "spreadsheet",
            "schema_description": schema_description,
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
