"""Deterministic Excel → DataFrame builder powered by SheetCompressor SABC.

Uses SpreadsheetLLM's Structural-Anchor-Based Compression (arXiv:2407.09025)
to detect headers, data columns, label columns, and data boundaries from
raw openpyxl cell metadata (font size, merge origin, borders, data types).

Then builds clean DataFrames from the detected structure.

Pipeline:
  openpyxl raw cells → SheetCompressor.analyze() → SheetStructure
                                                       ↓
                                              _build_df_from_structure()
                                                       ↓
                                                  clean DataFrame
"""

from __future__ import annotations

import re
import logging

import openpyxl
import pandas as pd

from spreadsheet_llm import SheetCompressor, SheetStructure

logger = logging.getLogger(__name__)


def build_dataframes(file_path: str) -> dict[str, pd.DataFrame]:
    """Build one clean DataFrame per visible sheet using SABC structure detection.

    Returns: {clean_sheet_name: DataFrame}
    """
    wb = openpyxl.load_workbook(file_path, data_only=True)
    compressor = SheetCompressor()
    dataframes = {}

    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        if ws.sheet_state != "visible":
            continue

        structure = compressor.analyze(ws)
        if structure is None:
            continue

        df = _build_df_from_structure(structure)
        if df is not None and not df.empty:
            clean_name = _clean_name(sheet_name)
            dataframes[clean_name] = df
            logger.info(
                "%s → %d rows × %d cols (headers: %s, labels: %s)",
                clean_name, len(df), len(df.columns),
                structure.header_rows, structure.label_cols,
            )

    wb.close()
    return dataframes


def _build_df_from_structure(structure: SheetStructure) -> pd.DataFrame | None:
    """Build a DataFrame from SheetCompressor's structural analysis.

    Reads values from the cell_grid using the detected coordinates.
    Builds hierarchical row labels using a section stack.
    """
    grid = structure.cell_grid
    data_cols = structure.data_cols
    label_cols = structure.label_cols
    data_start = structure.data_start_row
    data_end = structure.data_end_row
    col_names = structure.column_names

    # Section stack: tracks the current section label at each label column level.
    # e.g. stack[2] = "Operating expenses", stack[3] = "Cost of revenue"
    section_stack: dict[int, str] = {}
    rows = []

    for r in range(data_start, data_end + 1):
        # ── Read labels from each label column ──
        label_by_col: dict[int, str] = {}
        for lc in label_cols:
            cell = grid.get(r, {}).get(lc, {})
            val = cell.get("value")
            if val is not None and str(val).strip():
                label_by_col[lc] = str(val).strip()

        # ── Read data values ──
        values = []
        has_number = False
        for dc in data_cols:
            cell = grid.get(r, {}).get(dc, {})
            val = cell.get("value")
            if cell.get("type") == "n":
                has_number = True
            values.append(val)

        # ── Update section stack ──
        for lc in sorted(label_by_col.keys()):
            section_stack[lc] = label_by_col[lc]
            # A higher-level label clears deeper levels not in this row
            for deeper in label_cols:
                if deeper > lc and deeper not in label_by_col:
                    section_stack.pop(deeper, None)

        # Section header row (label but NO numbers) — update stack, skip as data
        if label_by_col and not has_number:
            continue
        if not has_number:
            continue

        # ── Build hierarchical label from section stack ──
        if label_by_col:
            deepest = max(label_by_col.keys())
            parts = []
            for lc in sorted(section_stack.keys()):
                if lc < deepest:
                    parts.append(section_stack[lc])
                elif lc == deepest:
                    parts.append(label_by_col[lc])
            full_label = " > ".join(parts) if parts else None
        else:
            parts = [
                section_stack[lc]
                for lc in sorted(section_stack.keys())
                if section_stack.get(lc)
            ]
            full_label = " > ".join(parts) if parts else None

        rows.append([full_label] + values)

    if not rows:
        return None

    # ── Build column names ──
    clean_cols = ["line_item"]
    for dc in data_cols:
        raw = col_names.get(dc, f"col_{dc}")
        clean = re.sub(r"[^a-zA-Z0-9_\s]", "", raw).strip()
        clean = re.sub(r"\s+", "_", clean).lower()
        if not clean:
            clean = f"col_{dc}"
        clean_cols.append(clean)

    # Deduplicate
    seen: dict[str, int] = {}
    deduped: list[str] = []
    for col in clean_cols:
        if col in seen:
            seen[col] += 1
            deduped.append(f"{col}_{seen[col]}")
        else:
            seen[col] = 0
            deduped.append(col)

    df = pd.DataFrame(rows, columns=deduped)

    # Convert numeric columns
    for col in deduped[1:]:
        try:
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() > 0:
                df[col] = converted
        except Exception:
            pass

    return df


def _clean_name(name: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9_]", "_", name.strip())
    clean = re.sub(r"_+", "_", clean).strip("_").lower()
    if not clean or clean[0].isdigit():
        clean = "t_" + clean
    return clean
