"""SpreadsheetLLM - SheetCompressor + Encoder Implementation.
Based on arXiv:2407.09025: "SpreadsheetLLM: Encoding Spreadsheets for Large Language Models"

Two-phase architecture:
  Phase 1: SheetCompressor.analyze()  — SABC on raw openpyxl worksheet
           Reads every cell's value, font, border, merge info to detect
           headers vs titles vs data vs labels. Returns a SheetStructure.

  Phase 2: SpreadsheetEncoder.encode() — DFAA + IIT encoding
           Takes a DataFrame + SheetStructure and outputs a compressed
           coordinate string for LLM consumption.

The SheetCompressor is the SINGLE source of structural truth.
Both the DataFrame builder and the chunk encoder consume its output.
"""

from __future__ import annotations

import re
import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


# ======================================================================
# Data structures
# ======================================================================

@dataclass
class SheetStructure:
    """Output of SABC structural analysis on a raw worksheet."""
    header_rows: list[int]           # 1-indexed rows detected as headers
    data_cols: list[int]             # 1-indexed columns with numeric data
    label_cols: list[int]            # 1-indexed columns with row labels
    data_start_row: int              # First row of actual data
    data_end_row: int                # Last row of actual data
    preamble_rows: list[int]         # Title/subtitle rows to skip
    summary_rows: list[int]          # Total/subtotal rows
    column_names: dict[int, str]     # data_col → constructed header name
    # Raw grid for downstream consumers (DataFrame builder, encoder)
    cell_grid: dict[int, dict[int, dict]]
    merge_map: dict[tuple, Any]      # (row, col) → anchor value
    merge_origins: dict[tuple, tuple] # (row, col) → (anchor_row, anchor_col)
    max_row: int = 0
    max_col: int = 0


# ======================================================================
# Phase 1: SheetCompressor (SABC)
# ======================================================================

class SheetCompressor:
    """Structural-Anchor-Based Compression on raw openpyxl worksheets.

    Analyzes a worksheet's raw cell grid using openpyxl metadata to detect:
    - Header rows  (scored using merge origin, font size, border, column coverage)
    - Data columns  (columns with >= MIN_DATA_COUNT non-year numbers)
    - Label columns (mostly-text columns left of first data column)
    - Data region   (start/end rows)
    - Summary rows  (totals, subtotals)

    No LLM. No guessing. Just math on openpyxl cell properties.
    """

    # Tuning knobs
    MIN_DATA_COUNT = 3           # Min non-year numbers to be a data column
    LABEL_NUMBER_RATIO = 0.3     # Max number ratio for label columns (0.3 catches footnote markers)
    DATA_ROW_THRESHOLD = 0.4     # Min fraction of data cols with numbers to start data
    HEADER_SCORE_THRESHOLD = 0.3 # Min score for a row to be classified as header
    HEADER_SCAN_WINDOW = 8       # How far above data_start to look for headers
    EMPTY_ROW_GAP = 3            # Consecutive empty rows = end of data
    YEAR_RANGE = (1900, 2100)    # Year-like number range
    FONT_TITLE_RATIO = 1.2      # Font size ratio above which = title, not header

    def analyze(self, ws) -> SheetStructure | None:
        """Run SABC on a worksheet. Returns SheetStructure or None."""
        max_row = ws.max_row or 0
        max_col = ws.max_column or 0
        if max_row < 2 or max_col < 2:
            return None

        # 1. Build enriched cell grid with openpyxl metadata
        cell_grid, merge_map, merge_origins = self._build_cell_grid(
            ws, max_row, max_col
        )

        # 2. Profile columns: count numbers vs text
        col_num, col_total = self._profile_columns(cell_grid, max_row, max_col)

        # 3. Data columns
        data_cols = sorted(
            c for c, count in col_num.items() if count >= self.MIN_DATA_COUNT
        )
        if not data_cols:
            return None
        first_data_col = data_cols[0]

        # 4. Label columns
        label_cols = self._find_label_cols(col_num, col_total, first_data_col)

        # 5. Data start row
        data_start = self._find_data_start(cell_grid, data_cols, max_row)
        if data_start is None:
            return None

        # 6. Median font size across all cells (titles are few, so median = data font)
        median_font = self._compute_median_font(cell_grid, max_row, max_col)

        # 7. Score and detect header rows
        header_rows = self._detect_headers(
            cell_grid, data_cols, first_data_col, data_start,
            merge_origins, median_font, max_col,
        )

        # 8. Column names from detected headers
        column_names = self._build_column_names(
            cell_grid, header_rows, data_cols, merge_map
        )

        # 9. Data end row
        data_end = self._find_data_end(
            cell_grid, data_cols, label_cols, data_start, max_row
        )

        # 10. Preamble = rows above first header (or data_start if no headers)
        cutoff = min(header_rows) if header_rows else data_start
        preamble_rows = list(range(1, cutoff))

        # 11. Summary rows
        summary_rows = self._detect_summary_rows(
            cell_grid, label_cols, data_start, data_end
        )

        return SheetStructure(
            header_rows=header_rows,
            data_cols=data_cols,
            label_cols=label_cols,
            data_start_row=data_start,
            data_end_row=data_end,
            preamble_rows=preamble_rows,
            summary_rows=summary_rows,
            column_names=column_names,
            cell_grid=cell_grid,
            merge_map=merge_map,
            merge_origins=merge_origins,
            max_row=max_row,
            max_col=max_col,
        )

    # ------------------------------------------------------------------
    # Cell grid builder
    # ------------------------------------------------------------------

    def _build_cell_grid(self, ws, max_row: int, max_col: int):
        """Read every cell with openpyxl metadata. Build merge maps."""
        cell_grid: dict[int, dict[int, dict]] = defaultdict(dict)
        merge_map: dict[tuple, Any] = {}
        merge_origins: dict[tuple, tuple] = {}

        # Merge maps
        for mr in ws.merged_cells.ranges:
            anchor_val = ws.cell(row=mr.min_row, column=mr.min_col).value
            for r in range(mr.min_row, mr.max_row + 1):
                for c in range(mr.min_col, mr.max_col + 1):
                    if (r, c) != (mr.min_row, mr.min_col):
                        merge_map[(r, c)] = anchor_val
                        merge_origins[(r, c)] = (mr.min_row, mr.min_col)

        # Read cells
        for row in ws.iter_rows(min_row=1, max_row=max_row, max_col=max_col):
            for cell in row:
                r, c = cell.row, cell.column
                val = cell.value
                if val is None and (r, c) in merge_map:
                    val = merge_map[(r, c)]

                # Data type
                dtype = None
                if val is None or str(val).strip() == "":
                    dtype = None
                elif isinstance(val, (int, float)):
                    dtype = "n"
                elif cell.data_type == "n":
                    dtype = "n"
                else:
                    dtype = "s"

                # Font size (default 11pt when not set)
                font_size = 11.0
                try:
                    if cell.font and cell.font.size is not None:
                        font_size = float(cell.font.size)
                except (TypeError, AttributeError):
                    pass

                # Bold
                font_bold = False
                try:
                    if cell.font and cell.font.bold:
                        font_bold = True
                except (TypeError, AttributeError):
                    pass

                # Bottom border
                border_bottom = None
                try:
                    if (cell.border and cell.border.bottom
                            and cell.border.bottom.style):
                        border_bottom = cell.border.bottom.style
                except (TypeError, AttributeError):
                    pass

                cell_grid[r][c] = {
                    "value": val,
                    "type": dtype,
                    "font_size": font_size,
                    "bold": font_bold,
                    "border_bottom": border_bottom,
                }

        return cell_grid, merge_map, merge_origins

    # ------------------------------------------------------------------
    # Column profiling
    # ------------------------------------------------------------------

    def _profile_columns(self, cell_grid, max_row, max_col):
        col_num: dict[int, int] = defaultdict(int)
        col_total: dict[int, int] = defaultdict(int)
        for r in range(1, max_row + 1):
            for c in range(1, max_col + 1):
                cell = cell_grid.get(r, {}).get(c, {})
                if cell.get("type") == "n":
                    val = cell["value"]
                    if isinstance(val, (int, float)) and self.YEAR_RANGE[0] <= val <= self.YEAR_RANGE[1]:
                        continue
                    col_num[c] += 1
                if cell.get("type") is not None:
                    col_total[c] += 1
        return col_num, col_total

    def _find_label_cols(self, col_num, col_total, first_data_col):
        label_cols = []
        for c in range(1, first_data_col):
            total = col_total.get(c, 0)
            if total == 0:
                continue
            if col_num.get(c, 0) / total < self.LABEL_NUMBER_RATIO:
                label_cols.append(c)
        if not label_cols and first_data_col > 1:
            label_cols = [first_data_col - 1]
        return label_cols

    # ------------------------------------------------------------------
    # Data region detection
    # ------------------------------------------------------------------

    def _find_data_start(self, cell_grid, data_cols, max_row):
        for r in range(1, max_row + 1):
            hits = 0
            for c in data_cols:
                cell = cell_grid.get(r, {}).get(c, {})
                if cell.get("type") == "n":
                    val = cell["value"]
                    if isinstance(val, (int, float)) and not (
                        self.YEAR_RANGE[0] <= val <= self.YEAR_RANGE[1]
                    ):
                        hits += 1
            if hits >= max(1, len(data_cols) * self.DATA_ROW_THRESHOLD):
                return r
        return None

    def _find_data_end(self, cell_grid, data_cols, label_cols, data_start, max_row):
        data_end = data_start
        for r in range(data_start, max_row + 1):
            has_any = False
            for c in data_cols:
                if cell_grid.get(r, {}).get(c, {}).get("type") == "n":
                    has_any = True
                    break
            if not has_any:
                for c in label_cols:
                    if cell_grid.get(r, {}).get(c, {}).get("type") == "s":
                        has_any = True
                        break
            if has_any:
                data_end = r
            elif r > data_end + self.EMPTY_ROW_GAP:
                break
        return data_end

    # ------------------------------------------------------------------
    # Header detection (the math)
    # ------------------------------------------------------------------

    def _compute_median_font(self, cell_grid, max_row, max_col) -> float:
        """Median font size across all non-empty cells.

        Titles are a few rows; data is many rows. So the median
        naturally equals the data font size without needing to know
        where data starts.
        """
        fonts = []
        for r in range(1, max_row + 1):
            for c in range(1, max_col + 1):
                cell = cell_grid.get(r, {}).get(c, {})
                if cell.get("type") is not None:
                    fonts.append(cell.get("font_size", 11.0))
        if not fonts:
            return 11.0
        fonts.sort()
        return fonts[len(fonts) // 2]

    def _score_header_row(
        self,
        row: int,
        cell_grid: dict,
        data_cols: list[int],
        first_data_col: int,
        merge_origins: dict,
        median_font: float,
        max_col: int,
    ) -> float:
        """Score a candidate row using 4 openpyxl-derived signals.

        Signal 1 — Data column coverage (0.50 weight)
            How many data columns have content ORIGINATING in the data
            region? A title merged from col A across everything scores 0.
            A header with text at each data column group scores high.

        Signal 2 — Year-like numbers (0.25 weight)
            Years (2023, 2024) at data column positions = strong header.

        Signal 3 — Font size ratio (±0.15/−0.30)
            Same font as data = header. Larger font = title → penalize.

        Signal 4 — Bottom border (0.10 weight)
            Border below a row = classic header separator.
        """
        score = 0.0

        # --- Signal 1: data column coverage ---
        data_origin_hits = 0
        for dc in data_cols:
            cell = cell_grid.get(row, {}).get(dc, {})
            val = cell.get("value")
            if val is None or str(val).strip() == "":
                continue
            if (row, dc) in merge_origins:
                _, anchor_col = merge_origins[(row, dc)]
                if anchor_col >= first_data_col:
                    data_origin_hits += 1
                # else: merge bleeds from label region → don't count
            else:
                data_origin_hits += 1

        coverage = data_origin_hits / len(data_cols) if data_cols else 0
        score += coverage * 0.50

        # --- Signal 2: year-like numbers at data columns ---
        for dc in data_cols:
            cell = cell_grid.get(row, {}).get(dc, {})
            val = cell.get("value")
            if isinstance(val, (int, float)) and self.YEAR_RANGE[0] <= val <= self.YEAR_RANGE[1]:
                score += 0.25
                break

        # --- Signal 3: font size ratio ---
        row_fonts = []
        for c in range(1, max_col + 1):
            cell = cell_grid.get(row, {}).get(c, {})
            if cell.get("type") is not None:
                row_fonts.append(cell.get("font_size", 11.0))

        if row_fonts and median_font > 0:
            avg_font = sum(row_fonts) / len(row_fonts)
            if avg_font / median_font > self.FONT_TITLE_RATIO:
                score -= 0.30   # big font = title
            else:
                score += 0.15   # normal font = consistent with header

        # --- Signal 4: bottom border ---
        for dc in data_cols:
            cell = cell_grid.get(row, {}).get(dc, {})
            if cell.get("border_bottom"):
                score += 0.10
                break

        return score

    def _detect_headers(
        self, cell_grid, data_cols, first_data_col, data_start,
        merge_origins, median_font, max_col,
    ) -> list[int]:
        """Score every candidate row above data_start, keep those above threshold."""
        header_rows = []
        scan_start = max(1, data_start - self.HEADER_SCAN_WINDOW)

        for r in range(scan_start, data_start):
            # Skip completely empty rows
            has_content = any(
                cell_grid.get(r, {}).get(c, {}).get("type") is not None
                for c in range(1, max_col + 1)
            )
            if not has_content:
                continue

            s = self._score_header_row(
                r, cell_grid, data_cols, first_data_col,
                merge_origins, median_font, max_col,
            )
            logger.debug("Row %d header score: %.2f", r, s)
            if s >= self.HEADER_SCORE_THRESHOLD:
                header_rows.append(r)

        return header_rows

    # ------------------------------------------------------------------
    # Column name construction
    # ------------------------------------------------------------------

    def _build_column_names(self, cell_grid, header_rows, data_cols, merge_map):
        column_names: dict[int, str] = {}
        for dc in data_cols:
            parts = []
            for hr in header_rows:
                val = None
                for try_col in [dc, dc - 1, dc + 1]:
                    cell = cell_grid.get(hr, {}).get(try_col, {})
                    v = cell.get("value")
                    if v is not None and str(v).strip():
                        val = str(v).strip()
                        break
                if val is None:
                    mv = merge_map.get((hr, dc))
                    if mv is not None:
                        val = str(mv).strip()
                if val:
                    parts.append(val)
            column_names[dc] = " | ".join(parts) if parts else f"col_{dc}"
        return column_names

    # ------------------------------------------------------------------
    # Summary row detection
    # ------------------------------------------------------------------

    def _detect_summary_rows(self, cell_grid, label_cols, data_start, data_end):
        pattern = re.compile(r"\b(total|sum|subtotal|grand\s*total|net)\b", re.I)
        rows = []
        for r in range(data_start, data_end + 1):
            for lc in label_cols:
                cell = cell_grid.get(r, {}).get(lc, {})
                val = cell.get("value")
                if val and pattern.search(str(val)):
                    rows.append(r)
                    break
        return rows


# ======================================================================
# Phase 2: SpreadsheetEncoder (DFAA + IIT)
# ======================================================================

@dataclass
class SpreadsheetLLMConfig:
    token_limit: int = 4000
    keep_first_rows: int = 3
    keep_last_rows: int = 2
    max_text_len: int = 100
    include_formats: bool = True


class SpreadsheetEncoder:
    """Encodes DataFrames into compressed coordinate strings for LLM consumption.

    Uses Data-Format-Aware Aggregation (DFAA) and Inverted-Index
    Translation (IIT) from the SpreadsheetLLM paper.
    """

    def __init__(self, config: SpreadsheetLLMConfig | None = None):
        self.config = config or SpreadsheetLLMConfig()

    def encode(
        self,
        df: pd.DataFrame,
        sheet_name: str,
        structure: SheetStructure | None = None,
    ) -> str:
        """Compress and encode a DataFrame into IIT format."""
        # 1. Anchor rows (from structure or heuristic fallback)
        anchor_rows = self._get_anchor_rows(df, structure)
        anchor_cols = list(range(len(df.columns)))

        # 2. Format map (DFAA)
        format_map = self._get_format_map(df, anchor_cols)

        # 3. Inverted-index translation (IIT)
        return self._translate(df, sheet_name, anchor_rows, anchor_cols, format_map)

    def _get_anchor_rows(self, df, structure):
        num_rows = len(df)
        anchors = set()
        # Boundaries
        for i in range(min(self.config.keep_first_rows, num_rows)):
            anchors.add(i)
        for i in range(max(0, num_rows - self.config.keep_last_rows), num_rows):
            anchors.add(i)
        # Summary rows from structure
        if structure:
            for sr in structure.summary_rows:
                idx = sr - structure.data_start_row
                if 0 <= idx < num_rows:
                    anchors.add(idx)
        # Keyword-based fallback
        kw = re.compile(r"(total|sum|average|subtotal|grand|net|gross)", re.I)
        for idx, row in df.iterrows():
            if kw.search(" ".join(str(v) for v in row.values)):
                anchors.add(idx)
        return sorted(anchors)

    def _get_format_map(self, df, anchor_cols):
        fmt = {}
        for ci in anchor_cols:
            col = df.columns[ci]
            fmt[ci] = "num" if pd.api.types.is_numeric_dtype(df[col].dtype) else "txt"
        return fmt

    def _translate(self, df, sheet_name, anchor_rows, anchor_cols, format_map):
        header_map = {i: col for i, col in enumerate(df.columns) if i in anchor_cols}

        fmt_block = ""
        if self.config.include_formats:
            entries = [f"C{i}:{f}" for i, f in format_map.items()]
            fmt_block = f"Formats:[{' '.join(entries)}] "

        frags = []
        for ri in anchor_rows:
            row_data = df.iloc[ri]
            for ci in anchor_cols:
                val = row_data.iloc[ci]
                if pd.isna(val) or val == "":
                    continue
                vs = str(val)
                if len(vs) > self.config.max_text_len:
                    vs = vs[: self.config.max_text_len] + "..."
                frags.append(f"R{ri}C{ci}:{vs}")

        return " | ".join([
            f"Sheet:{sheet_name}",
            f"Headers:{json.dumps(header_map)}",
            f"{fmt_block}Data:[{' '.join(frags)}]",
        ])


# ======================================================================
# Integration helper
# ======================================================================

def create_spreadsheet_llm_chunks(
    dataframes: Dict[str, pd.DataFrame],
    sheet_dna_or_structures: Any,
    filename: str,
    document_key: str,
) -> List:
    """Convert multiple sheets into SpreadsheetLLM-compressed chunks.

    Accepts either a dict of SheetDNA (legacy) or SheetStructure objects.
    """
    from chunking_pipeline import Chunk

    encoder = SpreadsheetEncoder()
    chunks = []

    for sheet_name, df in dataframes.items():
        # Try to find a matching structure/dna
        structure = None
        if isinstance(sheet_dna_or_structures, dict):
            target = re.sub(r"[^a-z0-9]+", "_", sheet_name.lower()).strip("_")
            for orig, obj in sheet_dna_or_structures.items():
                if re.sub(r"[^a-z0-9]+", "_", orig.lower()).strip("_") == target:
                    if isinstance(obj, SheetStructure):
                        structure = obj
                    break

        encoded = encoder.encode(df, sheet_name, structure)

        chunks.append(Chunk(
            text=encoded,
            metadata={
                "chunk_type": "spreadsheet_llm",
                "sheet_name": sheet_name,
                "document_name": filename,
                "document_key": document_key,
                "encoding_version": "v2.0-SABC-DFAA-IIT",
            },
        ))

    return chunks
