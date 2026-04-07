"""Pipeline stage debugger — dumps JSON artifacts between each stage.

When enabled (PIPELINE_DEBUG=1), writes JSON files to a per-file directory
so you can inspect exactly what data flows between stages:

    /tmp/excel_pipeline/<filename>/
        stage1_parse.json           — raw DataFrames as records + formulas + column dtypes
        stage1_5_classification.json — column roles + sheet summaries
        stage2_semantic_rows.json   — role-tagged semantic row strings
        stage2_schema.txt           — enriched schema description (plain text)
        stage3_chunks.json          — final chunks with metadata (before embedding)

Enable: export PIPELINE_DEBUG=1
Custom output dir: export PIPELINE_DEBUG_DIR=/path/to/dir
"""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


def is_debug_enabled() -> bool:
    return os.environ.get("PIPELINE_DEBUG", "").strip() in ("1", "true", "yes")


def _get_debug_dir(filename: str) -> Path:
    """Create and return the debug output directory for a given file."""
    base = Path(os.environ.get("PIPELINE_DEBUG_DIR", "/tmp/excel_pipeline"))
    # Sanitize filename for directory name
    safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", Path(filename).stem)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = base / f"{safe_name}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _to_serializable(obj: Any) -> Any:
    """Recursively convert numpy/pandas types to JSON-safe Python types."""
    if obj is None:
        return None
    if isinstance(obj, (str, int, bool)):
        return obj
    if isinstance(obj, float):
        if obj != obj:  # NaN
            return None
        return obj
    if hasattr(obj, "item"):  # numpy scalar
        return obj.item()
    if hasattr(obj, "isoformat"):  # datetime
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    return str(obj)


def _write_json(path: Path, data: Any) -> None:
    """Write JSON with pretty-printing, handling numpy/pandas types."""
    with open(path, "w") as f:
        json.dump(_to_serializable(data), f, indent=2, ensure_ascii=False)
    logger.info("[PipelineDebug] Wrote %s (%d bytes)", path.name, path.stat().st_size)


def _write_text(path: Path, text: str) -> None:
    """Write plain text file."""
    path.write_text(text, encoding="utf-8")
    logger.info("[PipelineDebug] Wrote %s (%d bytes)", path.name, path.stat().st_size)


class PipelineDebugger:
    """Collects and dumps per-stage pipeline artifacts to JSON files.

    Usage:
        debug = PipelineDebugger("budget.xlsx")
        debug.dump_stage1(dataframes, formulas)
        debug.dump_stage1_5(classification_meta)
        debug.dump_stage2(semantic_rows, schema_description)
        debug.dump_stage3(chunks)
        print(f"Debug output: {debug.output_dir}")

    Only writes files when PIPELINE_DEBUG=1 is set in environment.
    """

    def __init__(self, filename: str):
        self.filename = filename
        self.enabled = is_debug_enabled()
        self._output_dir: Path | None = None

    @property
    def output_dir(self) -> Path | None:
        return self._output_dir

    def dump_stage1(
        self,
        dataframes: dict[str, pd.DataFrame],
        formulas: dict[str, list[str]],
    ) -> None:
        """Dump Stage 1 (parsing) output: DataFrames + formulas."""
        if not self.enabled:
            return

        self._output_dir = _get_debug_dir(self.filename)

        stage1: dict[str, Any] = {
            "_stage": "1_parse",
            "_filename": self.filename,
            "_timestamp": datetime.now(timezone.utc).isoformat(),
            "sheets": {},
        }

        for sheet_name, df in dataframes.items():
            # Column metadata
            col_info = []
            for col in df.columns:
                if col == "__section__":
                    continue
                unique_vals = df[col].dropna().unique()
                info: dict[str, Any] = {
                    "name": col,
                    "dtype": str(df[col].dtype),
                    "non_null": int(df[col].notna().sum()),
                    "total": len(df),
                    "unique_count": len(unique_vals),
                    "sample_values": unique_vals[:5].tolist(),
                }
                if df[col].dtype.kind in ("i", "f"):
                    info["min"] = df[col].min()
                    info["max"] = df[col].max()
                col_info.append(info)

            # DataFrame as records (row-oriented JSON)
            records = df.to_dict(orient="records")

            sheet_formulas = formulas.get(sheet_name, [])

            stage1["sheets"][sheet_name] = {
                "row_count": len(df),
                "column_count": len([c for c in df.columns if c != "__section__"]),
                "columns": col_info,
                "formulas": sheet_formulas,
                "data": records,
            }

        _write_json(self._output_dir / "stage1_parse.json", stage1)
        print(f"[PipelineDebug] Stage 1 → {self._output_dir / 'stage1_parse.json'}")

    def dump_stage1_5(
        self,
        classification_meta: dict[str, dict],
    ) -> None:
        """Dump Stage 1.5 (classification) output: column roles + sheet summaries."""
        if not self.enabled or not self._output_dir:
            return

        stage1_5: dict[str, Any] = {
            "_stage": "1.5_classification",
            "_filename": self.filename,
            "_timestamp": datetime.now(timezone.utc).isoformat(),
            "sheets": classification_meta,
        }

        _write_json(self._output_dir / "stage1_5_classification.json", stage1_5)
        print(f"[PipelineDebug] Stage 1.5 → {self._output_dir / 'stage1_5_classification.json'}")

    def dump_stage2(
        self,
        semantic_rows: list[dict],
        schema_description: str,
    ) -> None:
        """Dump Stage 2 (enrichment) output: semantic rows + schema text."""
        if not self.enabled or not self._output_dir:
            return

        stage2: dict[str, Any] = {
            "_stage": "2_enrichment",
            "_filename": self.filename,
            "_timestamp": datetime.now(timezone.utc).isoformat(),
            "total_semantic_rows": len(semantic_rows),
            "semantic_rows": semantic_rows,
        }

        _write_json(self._output_dir / "stage2_semantic_rows.json", stage2)
        _write_text(self._output_dir / "stage2_schema.txt", schema_description)
        print(f"[PipelineDebug] Stage 2 → {self._output_dir / 'stage2_semantic_rows.json'}")
        print(f"[PipelineDebug] Stage 2 → {self._output_dir / 'stage2_schema.txt'}")

    def dump_stage3(
        self,
        chunks: list,
    ) -> None:
        """Dump Stage 3 (chunking) output: final chunks with metadata."""
        if not self.enabled or not self._output_dir:
            return

        chunk_list = []
        for i, chunk in enumerate(chunks):
            entry: dict[str, Any] = {
                "index": i,
                "text": chunk.text[:500],  # truncate long text
                "text_length": len(chunk.text),
            }
            if hasattr(chunk, "metadata") and chunk.metadata:
                # Copy metadata but skip huge fields
                meta = dict(chunk.metadata)
                if "schema_description" in meta:
                    meta["schema_description"] = f"({len(meta['schema_description'])} chars — see stage2_schema.txt)"
                if "sheet_summaries" in meta:
                    meta["sheet_summaries"] = "(see stage1_5_classification.json)"
                entry["metadata"] = meta
            chunk_list.append(entry)

        stage3: dict[str, Any] = {
            "_stage": "3_chunking",
            "_filename": self.filename,
            "_timestamp": datetime.now(timezone.utc).isoformat(),
            "total_chunks": len(chunks),
            "chunks": chunk_list,
        }

        _write_json(self._output_dir / "stage3_chunks.json", stage3)
        print(f"[PipelineDebug] Stage 3 → {self._output_dir / 'stage3_chunks.json'}")

    def print_summary(self) -> None:
        """Print a summary of all dumped files."""
        if not self.enabled or not self._output_dir:
            return

        print(f"\n{'='*60}")
        print(f"  PIPELINE DEBUG OUTPUT: {self.filename}")
        print(f"  Directory: {self._output_dir}")
        print(f"{'='*60}")
        for f in sorted(self._output_dir.iterdir()):
            size_kb = f.stat().st_size / 1024
            print(f"  {f.name:40s} {size_kb:8.1f} KB")
        print(f"{'='*60}\n")
