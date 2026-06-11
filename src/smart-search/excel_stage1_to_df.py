"""Build clean DataFrames from Stage 1 ExtractionCell data.

Uses the provenance that openpyxl + our enrichment already computed:
  - cell.row_label → "Sales and marketing"
  - cell.section_path → ["Operating expenses"]
  - cell.column_header_path → ["Three Months Ended", "June 30", "2003(1)"]
  - cell.unit_context → "(In millions)"
  - cell.raw_value → 2288

No LLM needed. The schema comes from the data itself.

Usage:
    from excel_stage1_to_df import build_dataframes_from_stage1
    dataframes = build_dataframes_from_stage1(stage1_result)
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict

import pandas as pd

logger = logging.getLogger(__name__)


def build_dataframes_from_stage1(stage1_result) -> dict[str, pd.DataFrame]:
    """Convert Stage 1 ExtractionCells into clean DataFrames for SQLite.

    Groups cells by sheet → builds one DataFrame per sheet with:
      - line_item column: section_path + row_label (handles duplicates)
      - data columns: flattened column_header_path
    """
    if not stage1_result or not stage1_result.cells_by_sheet:
        return {}

    dataframes = {}

    for sheet_name, cells in stage1_result.cells_by_sheet.items():
        df = _build_sheet_dataframe(sheet_name, cells)
        if df is not None and not df.empty:
            clean_name = _clean_name(sheet_name)
            dataframes[clean_name] = df
            logger.info(
                "Stage1→DataFrame: %s → %d rows × %d cols",
                clean_name, len(df), len(df.columns),
            )

    return dataframes


def _build_sheet_dataframe(sheet_name: str, cells: list) -> pd.DataFrame | None:
    """Build a single DataFrame from a sheet's ExtractionCells."""

    # Step 1: Find all unique column headers and row labels
    # Group cells by (row, column_header_path)
    rows_data: dict[int, dict] = defaultdict(dict)
    row_labels: dict[int, str] = {}
    column_headers: dict[str, str] = {}  # col_path_key → clean_name

    for cell in cells:
        # Skip cells without numeric data
        if cell.data_type not in ("n", "f") or cell.raw_value is None:
            continue

        # Skip if no column header (can't place this value)
        if not cell.column_header_path:
            continue

        row = cell.row

        # Build line_item: section_path + row_label
        if row not in row_labels and cell.row_label:
            parts = []
            if cell.section_path:
                parts.extend(cell.section_path)
            parts.append(cell.row_label)
            row_labels[row] = " > ".join(parts)

        # Build column name from header path
        col_key = " | ".join(cell.column_header_path)
        if col_key not in column_headers:
            column_headers[col_key] = _clean_column_name(col_key)

        # Store the value
        rows_data[row][col_key] = cell.raw_value

    if not rows_data or not column_headers:
        return None

    # Step 2: Also capture rows that are ONLY labels (section headers with no data)
    for cell in cells:
        if cell.data_type == "s" and cell.row_label and cell.row not in row_labels:
            if cell.section_path:
                row_labels[cell.row] = " > ".join(cell.section_path + [cell.row_label])
            else:
                row_labels[cell.row] = cell.row_label

    # Step 3: Build the DataFrame
    sorted_rows = sorted(rows_data.keys())
    sorted_col_keys = sorted(column_headers.keys(), key=lambda k: list(column_headers.keys()).index(k))

    records = []
    for row in sorted_rows:
        label = row_labels.get(row)
        values = rows_data[row]
        record = {"line_item": label}
        for col_key in sorted_col_keys:
            clean_col = column_headers[col_key]
            record[clean_col] = values.get(col_key)
        records.append(record)

    if not records:
        return None

    df = pd.DataFrame(records)

    # Deduplicate column names
    cols = list(df.columns)
    seen: dict[str, int] = {}
    deduped = []
    for col in cols:
        if col in seen:
            seen[col] += 1
            deduped.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            deduped.append(col)
    df.columns = deduped

    # Convert numeric columns
    for col in df.columns:
        if col == "line_item":
            continue
        try:
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() > 0:
                df[col] = converted
        except Exception:
            pass

    return df


def _clean_column_name(col_path: str) -> str:
    """Flatten column header path into SQL-friendly name."""
    # "Three Months Ended | June 30 | 2003(1)" → "three_months_ended_june_30_2003_1"
    clean = col_path.lower()
    clean = re.sub(r"[^a-z0-9]+", "_", clean)
    clean = re.sub(r"_+", "_", clean).strip("_")
    if not clean:
        clean = "col"
    if clean[0].isdigit():
        clean = "c_" + clean
    return clean


def _clean_name(name: str) -> str:
    """Clean sheet name for use as table name."""
    clean = re.sub(r"[^a-zA-Z0-9_]", "_", name.strip())
    clean = re.sub(r"_+", "_", clean).strip("_").lower()
    if not clean or clean[0].isdigit():
        clean = "t_" + clean
    return clean
