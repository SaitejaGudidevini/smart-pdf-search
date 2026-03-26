"""FastAPI routes for async Excel file indexing via Celery.

Endpoints
---------
POST /api/excel/upload   — Upload an Excel file and dispatch background indexing.
GET  /api/excel/status/{task_id} — Poll progress of a running task.
POST /api/excel/retry/{task_id}  — Retry a failed task.

Usage: register this router in app.py (or wherever the FastAPI app is defined):

    from excel_routes import router as excel_router
    app.include_router(excel_router)
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, HTTPException, UploadFile, File

from excel_parser import is_excel_file

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/excel", tags=["excel"])

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Maximum upload size: 100 MB
MAX_UPLOAD_BYTES = 100 * 1024 * 1024


def _get_celery_app():
    """Lazily import Celery app to avoid import-time broker connection."""
    from celery_config import app
    return app


def _get_async_result(task_id: str):
    """Return a Celery AsyncResult for the given task ID."""
    celery_app = _get_celery_app()
    return celery_app.AsyncResult(task_id)


def _get_redis_client():
    """Return a Redis client connected to the result backend."""
    import os
    import redis as redis_lib

    redis_url = os.environ.get("CELERY_RESULT_BACKEND", "redis://redis:6379/2")
    return redis_lib.from_url(redis_url)


def _store_task_args(task_id: str, kwargs: dict) -> None:
    """Persist task kwargs in Redis so failed tasks can be retried."""
    import json

    client = _get_redis_client()
    client.setex(
        f"excel:task_args:{task_id}",
        86400,  # 24-hour TTL
        json.dumps(kwargs),
    )


# ------------------------------------------------------------------
# POST /api/excel/upload
# ------------------------------------------------------------------


@router.post("/upload")
async def excel_upload(
    file: UploadFile = File(...),
    mayan_doc_id: int | None = None,
):
    """Upload an Excel file and start background indexing.

    Returns immediately with a ``task_id`` that can be polled via
    ``GET /api/excel/status/{task_id}``.

    Parameters
    ----------
    file:
        The Excel file (.xlsx, .xls, .csv).
    mayan_doc_id:
        Optional Mayan EDMS document ID for cross-referencing.

    Returns
    -------
    JSON with task_id, filename, and status.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    if not is_excel_file(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Expected .xlsx, .xls, or .csv — got '{file.filename}'.",
        )

    # Read file content with size check
    content = await file.read()
    if len(content) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(content) / 1024 / 1024:.1f} MB). Maximum is {MAX_UPLOAD_BYTES / 1024 / 1024:.0f} MB.",
        )

    # Save to disk with a unique name to prevent collisions
    safe_name = f"{uuid.uuid4().hex}_{file.filename}"
    dest = UPLOAD_DIR / safe_name
    dest.write_bytes(content)

    logger.info(
        "Received Excel upload: %s (%d bytes), saved as %s",
        file.filename,
        len(content),
        dest,
    )

    # Dispatch Celery task
    from excel_worker import index_excel_task

    task_kwargs = {
        "file_path": str(dest.resolve()),
        "filename": file.filename,
        "mayan_doc_id": mayan_doc_id,
    }

    task = index_excel_task.apply_async(kwargs=task_kwargs, queue="excel")

    # Persist kwargs in Redis so /retry can re-dispatch without re-upload
    _store_task_args(task.id, task_kwargs)

    return {
        "task_id": task.id,
        "filename": file.filename,
        "status": "queued",
        "message": "File uploaded. Indexing has been queued for background processing.",
    }


# ------------------------------------------------------------------
# GET /api/excel/status/{task_id}
# ------------------------------------------------------------------


@router.get("/status/{task_id}")
async def excel_status(task_id: str):
    """Poll the progress of a background Excel indexing task.

    Returns
    -------
    JSON with status, progress_pct, stage, message, and (on success) result.
    """
    result = _get_async_result(task_id)

    if result.state == "PENDING":
        return {
            "task_id": task_id,
            "status": "pending",
            "progress_pct": 0,
            "stage": "queued",
            "message": "Task is waiting in the queue.",
        }

    if result.state == "STARTED":
        return {
            "task_id": task_id,
            "status": "started",
            "progress_pct": 5,
            "stage": "initializing",
            "message": "Worker has picked up the task.",
        }

    if result.state == "PROGRESS":
        meta = result.info or {}
        return {
            "task_id": task_id,
            "status": "processing",
            "progress_pct": meta.get("progress", 0),
            "stage": meta.get("stage", "unknown"),
            "message": meta.get("message", "Processing..."),
        }

    if result.state == "SUCCESS":
        return {
            "task_id": task_id,
            "status": "completed",
            "progress_pct": 100,
            "stage": "done",
            "message": "Indexing completed successfully.",
            "result": result.result,
        }

    if result.state == "FAILURE":
        # result.info may be an exception or a dict from update_state
        error_info = result.info
        if isinstance(error_info, dict):
            message = error_info.get("message", "Task failed.")
        elif isinstance(error_info, Exception):
            message = str(error_info)
        else:
            message = "Task failed with an unknown error."

        return {
            "task_id": task_id,
            "status": "failed",
            "progress_pct": 0,
            "stage": "error",
            "message": message,
        }

    if result.state == "RETRY":
        meta = result.info or {}
        return {
            "task_id": task_id,
            "status": "retrying",
            "progress_pct": 0,
            "stage": "retry",
            "message": f"Task is being retried. Reason: {meta}" if meta else "Task is being retried.",
        }

    # REVOKED or other unknown states
    return {
        "task_id": task_id,
        "status": result.state.lower(),
        "progress_pct": 0,
        "stage": "unknown",
        "message": f"Task is in state: {result.state}",
    }


# ------------------------------------------------------------------
# POST /api/excel/retry/{task_id}
# ------------------------------------------------------------------


@router.post("/retry/{task_id}")
async def excel_retry(task_id: str):
    """Retry a failed Excel indexing task.

    Fetches the original arguments from the failed task result and
    dispatches a new task with the same parameters.

    Returns
    -------
    JSON with the new task_id and status.
    """
    result = _get_async_result(task_id)

    # Only allow retrying tasks that have finished (failed or succeeded)
    if result.state in ("PENDING", "STARTED", "PROGRESS", "RETRY"):
        raise HTTPException(
            status_code=409,
            detail=f"Task is still running (state={result.state}). Cannot retry.",
        )

    # Retrieve the original task kwargs persisted at upload time
    import json

    redis_client = _get_redis_client()
    stored_raw = redis_client.get(f"excel:task_args:{task_id}")

    if stored_raw is None:
        raise HTTPException(
            status_code=404,
            detail="Original task arguments not found. Cannot retry — please re-upload the file.",
        )

    original_kwargs = json.loads(stored_raw)

    # Verify the file still exists on disk
    file_path = original_kwargs.get("file_path", "")
    if not Path(file_path).exists():
        raise HTTPException(
            status_code=404,
            detail=f"Original file no longer exists at {file_path}. Please re-upload.",
        )

    # Dispatch a new task with the same arguments
    from excel_worker import index_excel_task

    new_task = index_excel_task.apply_async(
        kwargs=original_kwargs,
        queue="excel",
    )

    # Persist kwargs for the new task so it can also be retried later
    _store_task_args(new_task.id, original_kwargs)

    return {
        "original_task_id": task_id,
        "new_task_id": new_task.id,
        "status": "queued",
        "message": "Task has been re-queued for background processing.",
    }
