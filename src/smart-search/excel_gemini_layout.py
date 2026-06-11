"""Gemini-powered Excel layout detection + clean DataFrame builder.

Sends the cell grid to Gemini which returns the table structure as JSON.
Then openpyxl reads actual cell values using Gemini's layout instructions.
Result: ONE clean DataFrame per sheet → ONE clean SQLite table.

Usage:
    detector = GeminiLayoutDetector()
    dataframes = detector.parse_workbook("Financial Statements.xlsx")
    # Returns: {"income_statements": df1, "balance_sheets": df2, ...}
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import openpyxl
import pandas as pd

logger = logging.getLogger(__name__)

_GEMINI_AVAILABLE = False
try:
    from google import genai as _genai
    _GEMINI_AVAILABLE = True
except ImportError:
    pass

_LAYOUT_PROMPT = """\
You are an Excel spreadsheet analyst. Analyze this cell grid and return the table structure as JSON.

CELL GRID:
{cell_grid}

Return a JSON object with these fields:
{{
  "sheet_title": "human readable name (e.g. Income Statements)",
  "units": "the units line if present (e.g. In millions) or null",
  "header_rows": [list of 1-based row numbers that form the column headers],
  "data_start_row": first 1-based row number where actual data begins,
  "data_end_row": last 1-based row number with data (before footnotes),
  "label_columns": ["letters of ALL columns that contain row labels — financial statements often use MULTIPLE columns for hierarchy: e.g. B for section headers like 'Operating expenses:', C for sub-items like 'Cost of revenue', D for totals like 'Total operating expenses'. List ALL of them."],
  "data_columns": ["letters of columns containing numeric data (e.g. F, H, J)"],
  "column_names": ["flattened header for each data column, joining multi-row headers with ' | '"],
  "sections": ["list of section names found in the data, e.g. Operating expenses, Assets"]
}}

Rules:
- header_rows are rows with text that describe the data columns (like time periods, not the company name or title)
- Title rows (company name, sheet name) are NOT header rows
- Unit rows like "(In millions)" are NOT header rows
- data_columns are ONLY columns that contain numbers, not label columns
- column_names must have exactly the same count as data_columns, in the same order
- Flatten multi-row headers by joining with ' | ' (e.g. "Three Months Ended | June 30 | 2003")
- IMPORTANT: row labels often span MULTIPLE columns. Look at ALL text columns left of the data. If "Operating expenses:" is in col B and "Cost of revenue" is in col C, include BOTH B and C in label_columns
- If a header spans multiple columns, repeat it for each data column it covers
- Return ONLY the JSON object, no explanation"""


class GeminiLayoutDetector:
    """Use Gemini to detect Excel table layout, then build clean DataFrames."""

    def __init__(self):
        self._google_key = os.environ.get("GOOGLE_API_KEY")
        self._groq_key = os.environ.get("GROQ_API_KEY")

    def parse_workbook(self, file_path: str) -> dict[str, pd.DataFrame]:
        """Parse all visible sheets into clean DataFrames.

        Returns: {clean_sheet_name: DataFrame}
        """
        if not self._google_key and not self._groq_key:
            logger.warning("No LLM API key available — cannot detect layout")
            return {}

        ext = Path(file_path).suffix.lower()

        if ext == ".csv":
            return self._parse_csv(file_path)

        # Convert .xls to .xlsx so openpyxl can read it with full metadata
        actual_path = file_path
        if ext == ".xls":
            actual_path = self._convert_xls_to_xlsx(file_path)
            if not actual_path:
                return self._parse_xls_workbook(file_path)  # fallback to xlrd

        wb = openpyxl.load_workbook(actual_path, data_only=True)
        dataframes = {}

        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            if ws.sheet_state != "visible":
                continue

            df = self._parse_sheet(ws, file_path)
            if df is not None and not df.empty:
                clean_name = self._clean_name(sheet_name)
                dataframes[clean_name] = df
                logger.info("Gemini layout: %s → %d rows × %d cols", clean_name, len(df), len(df.columns))

        wb.close()
        return dataframes

    def _convert_xls_to_xlsx(self, file_path: str) -> str | None:
        """Convert .xls to .xlsx using LibreOffice (preserves everything).

        LibreOffice does a true format conversion — merged cells, fonts,
        colors, formulas, comments, number formats all transfer to .xlsx.
        Then openpyxl reads the full metadata.
        """
        import subprocess
        import shutil

        lo = shutil.which("libreoffice") or shutil.which("soffice")
        if not lo:
            logger.warning("LibreOffice not installed — cannot convert .xls to .xlsx")
            return None

        try:
            out_dir = str(Path(file_path).parent)
            result = subprocess.run(
                [lo, "--headless", "--convert-to", "xlsx", file_path, "--outdir", out_dir],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                logger.warning("LibreOffice conversion failed: %s", result.stderr[:200])
                return None

            xlsx_path = file_path.rsplit(".", 1)[0] + ".xlsx"
            if Path(xlsx_path).exists():
                logger.info("Converted .xls → .xlsx (LibreOffice): %s", xlsx_path)
                return xlsx_path

            logger.warning("LibreOffice ran but .xlsx not found at %s", xlsx_path)
            return None
        except subprocess.TimeoutExpired:
            logger.warning("LibreOffice conversion timed out")
            return None
        except Exception as e:
            logger.warning("LibreOffice conversion error: %s", e)
            return None

    def _parse_xls_workbook(self, file_path: str) -> dict[str, pd.DataFrame]:
        """Parse legacy .xls files using xlrd via pandas + LLM layout detection."""
        try:
            sheets = pd.read_excel(file_path, sheet_name=None, header=None, engine="xlrd")
        except Exception as e:
            logger.error("Failed to read .xls file: %s", e)
            return {}

        dataframes = {}
        for sheet_name, raw_df in sheets.items():
            # Build cell grid from the raw DataFrame
            lines = []
            for r_idx, row in raw_df.iterrows():
                for c_idx, val in enumerate(row):
                    if pd.notna(val) and str(val).strip():
                        col_letter = chr(65 + c_idx) if c_idx < 26 else f"{chr(64 + c_idx // 26)}{chr(65 + c_idx % 26)}"
                        lines.append(f"{col_letter}{r_idx + 1}: {val}")
            cell_grid = "\n".join(lines)[:8000]

            if not cell_grid.strip():
                continue

            layout = self._detect_layout(cell_grid)
            if not layout:
                continue

            # For .xls, build DataFrame from the raw pandas data using layout info
            df = self._build_dataframe_from_raw(raw_df, layout)
            if df is not None and not df.empty:
                clean_name = self._clean_name(sheet_name)
                dataframes[clean_name] = df
                logger.info("XLS layout: %s → %d rows × %d cols", clean_name, len(df), len(df.columns))

        return dataframes

    def _build_dataframe_from_raw(self, raw_df: pd.DataFrame, layout: dict) -> pd.DataFrame | None:
        """Build clean DataFrame from raw pandas DataFrame using LLM layout."""
        data_start = layout["data_start_row"] - 1  # 0-indexed for pandas
        data_end = (layout.get("data_end_row") or len(raw_df)) - 1
        label_col_idx = self._col_letter_to_idx(layout["label_column"]) - 1
        data_col_indices = [self._col_letter_to_idx(c) - 1 for c in layout["data_columns"]]
        column_names = layout["column_names"]

        rows = []
        for r in range(data_start, min(data_end + 1, len(raw_df))):
            label = raw_df.iloc[r, label_col_idx] if label_col_idx < len(raw_df.columns) else None
            if pd.notna(label):
                label = str(label).strip()
            else:
                label = None
            values = [raw_df.iloc[r, c] if c < len(raw_df.columns) else None for c in data_col_indices]
            if label is None and all(pd.isna(v) if not isinstance(v, str) else False for v in values):
                continue
            rows.append([label] + [v if pd.notna(v) else None for v in values])

        if not rows:
            return None

        all_columns = ["line_item"] + column_names
        clean_columns = []
        for col in all_columns:
            clean = re.sub(r"[^a-zA-Z0-9_\s]", "", str(col)).strip()
            clean = re.sub(r"\s+", "_", clean).lower()
            if not clean:
                clean = f"col_{len(clean_columns)}"
            clean_columns.append(clean)

        seen: dict[str, int] = {}
        deduped = []
        for col in clean_columns:
            if col in seen:
                seen[col] += 1
                deduped.append(f"{col}_{seen[col]}")
            else:
                seen[col] = 0
                deduped.append(col)

        df = pd.DataFrame(rows, columns=deduped)
        for col in deduped[1:]:
            try:
                converted = pd.to_numeric(df[col], errors="coerce")
                if converted.notna().sum() > 0:
                    df[col] = converted
            except Exception:
                pass
        return df

    def _parse_csv(self, file_path: str) -> dict[str, pd.DataFrame]:
        """Parse CSV files — simple, just read as-is."""
        try:
            df = pd.read_csv(file_path)
            clean_cols = [re.sub(r"[^a-zA-Z0-9_]", "_", str(c)).lower().strip("_") for c in df.columns]
            df.columns = clean_cols
            name = self._clean_name(Path(file_path).stem)
            return {name: df}
        except Exception as e:
            logger.error("Failed to read CSV: %s", e)
            return {}

    def _parse_sheet(self, ws, file_path: str) -> pd.DataFrame | None:
        """Parse a single sheet using Gemini layout detection."""
        # Build cell grid (same format we already use for Gemini)
        cell_grid = self._build_cell_grid(ws)
        if not cell_grid.strip():
            return None

        # Ask Gemini for the layout
        layout = self._detect_layout(cell_grid)
        if not layout:
            return None

        # Build DataFrame using the layout
        return self._build_dataframe(ws, layout)

    def _build_cell_grid(self, ws) -> str:
        """Build cell coordinate grid for Gemini."""
        lines = []
        max_row = min(ws.max_row or 0, 50)  # limit to first 50 rows
        for row in ws.iter_rows(min_row=1, max_row=max_row):
            for cell in row:
                if cell.value is not None and str(cell.value).strip():
                    lines.append(f"{cell.coordinate}: {cell.value}")
        return "\n".join(lines)[:8000]  # limit to 8K chars

    def _detect_layout(self, cell_grid: str) -> dict | None:
        """Send cell grid to Gemini and get table structure JSON."""
        import time

        prompt = _LAYOUT_PROMPT.format(cell_grid=cell_grid)
        raw = None

        # Try Groq first (fast, reliable)
        if self._groq_key:
            raw = self._call_groq(prompt)

        # Fallback to Gemini
        if not raw and self._google_key and _GEMINI_AVAILABLE:
            raw = self._call_gemini(prompt)

        if not raw:
            logger.error("Layout detection failed — no LLM responded")
            return None

        # Parse JSON from response
        parsed = self._parse_json(raw)
        if not isinstance(parsed, dict):
            logger.warning("Gemini returned non-dict: %s", raw[:200])
            return None

        # Validate required fields
        # Support both old "label_column" and new "label_columns"
        if "label_columns" in parsed and "label_column" not in parsed:
            parsed["label_column"] = parsed["label_columns"]
        elif "label_column" in parsed and "label_columns" not in parsed:
            lc = parsed["label_column"]
            parsed["label_columns"] = [lc] if isinstance(lc, str) else lc

        required = ["header_rows", "data_start_row", "data_columns", "column_names"]
        for field in required:
            if field not in parsed:
                logger.warning("Gemini missing field '%s' in response", field)
                return None

        if len(parsed["data_columns"]) != len(parsed["column_names"]):
            logger.warning(
                "Gemini column mismatch: %d data_columns vs %d column_names",
                len(parsed["data_columns"]), len(parsed["column_names"]),
            )
            # Try to fix by truncating to shorter length
            min_len = min(len(parsed["data_columns"]), len(parsed["column_names"]))
            parsed["data_columns"] = parsed["data_columns"][:min_len]
            parsed["column_names"] = parsed["column_names"][:min_len]

        logger.info(
            "Gemini layout: headers=%s, data=%d-%s, labels=%s, %d data cols",
            parsed["header_rows"], parsed["data_start_row"],
            parsed.get("data_end_row", "?"), parsed["label_column"],
            len(parsed["data_columns"]),
        )
        return parsed

    def _build_dataframe(self, ws, layout: dict) -> pd.DataFrame | None:
        """Build a clean DataFrame from openpyxl cells using Gemini's layout."""
        data_start = layout["data_start_row"]
        data_end = layout.get("data_end_row") or ws.max_row or data_start
        label_col_letters = layout.get("label_columns") or [layout.get("label_column", "B")]
        if isinstance(label_col_letters, str):
            label_col_letters = [label_col_letters]
        data_col_letters = layout["data_columns"]
        column_names = layout["column_names"]

        # Convert column letters to 1-based indices
        label_col_indices = [self._col_letter_to_idx(c) for c in label_col_letters]
        data_col_indices = [self._col_letter_to_idx(c) for c in data_col_letters]

        # Read data rows
        rows = []
        for row_num in range(data_start, data_end + 1):
            # Get row label from ALL label columns (join non-empty ones)
            label_parts = []
            for lci in label_col_indices:
                cell = ws.cell(row=row_num, column=lci)
                if cell.value is not None:
                    text = str(cell.value).strip()
                    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text).strip()
                    if text:
                        label_parts.append(text)
            label = " > ".join(label_parts) if label_parts else None

            # Get data values
            values = []
            for col_idx in data_col_indices:
                cell = ws.cell(row=row_num, column=col_idx)
                val = cell.value
                # Clean error values
                if isinstance(val, str) and val.startswith("#"):
                    val = None
                values.append(val)

            # Skip completely empty rows
            if label is None and all(v is None for v in values):
                continue

            rows.append([label] + values)

        if not rows:
            return None

        # Build DataFrame
        all_columns = ["line_item"] + column_names

        # Clean column names for SQL compatibility
        clean_columns = []
        for col in all_columns:
            clean = re.sub(r"[^a-zA-Z0-9_\s]", "", str(col)).strip()
            clean = re.sub(r"\s+", "_", clean).lower()
            if not clean:
                clean = f"col_{len(clean_columns)}"
            clean_columns.append(clean)

        # Deduplicate column names
        seen: dict[str, int] = {}
        deduped = []
        for col in clean_columns:
            if col in seen:
                seen[col] += 1
                deduped.append(f"{col}_{seen[col]}")
            else:
                seen[col] = 0
                deduped.append(col)

        df = pd.DataFrame(rows, columns=deduped)

        # Try to convert numeric columns
        for col in deduped[1:]:  # skip line_item
            try:
                converted = pd.to_numeric(df[col], errors="coerce")
                if converted.notna().sum() > 0:
                    df[col] = converted
            except Exception:
                pass

        # Add metadata
        df.attrs["units"] = layout.get("units")
        df.attrs["sheet_title"] = layout.get("sheet_title")
        df.attrs["sections"] = layout.get("sections", [])

        return df

    @staticmethod
    def _col_letter_to_idx(letter: str) -> int:
        """Convert Excel column letter to 1-based index. A=1, B=2, Z=26, AA=27."""
        result = 0
        for c in letter.upper():
            result = result * 26 + (ord(c) - ord("A") + 1)
        return result

    @staticmethod
    def _clean_name(name: str) -> str:
        """Clean sheet name for use as table name."""
        clean = re.sub(r"[^a-zA-Z0-9_]", "_", name.strip())
        clean = re.sub(r"_+", "_", clean).strip("_").lower()
        if not clean or clean[0].isdigit():
            clean = "t_" + clean
        return clean

    @staticmethod
    def _parse_json(raw: str) -> Any:
        """Extract JSON from Gemini response."""
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            text = "\n".join(lines).strip()

        try:
            return json.loads(text)
        except json.JSONDecodeError:
            for pattern in [r"\{[\s\S]*\}"]:
                match = re.search(pattern, text)
                if match:
                    try:
                        return json.loads(match.group())
                    except json.JSONDecodeError:
                        continue
        return None

    def _call_groq(self, prompt: str) -> str | None:
        """Call Groq API for layout detection."""
        import httpx
        try:
            resp = httpx.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {self._groq_key}", "Content-Type": "application/json"},
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [
                        {"role": "system", "content": "You are an Excel spreadsheet analyst. Return only valid JSON."},
                        {"role": "user", "content": prompt},
                    ],
                    "max_tokens": 1000, "temperature": 0,
                },
                timeout=30,
            )
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
            logger.warning("Groq returned %d", resp.status_code)
        except Exception as e:
            logger.warning("Groq layout call failed: %s", e)
        return None

    def _call_gemini(self, prompt: str) -> str | None:
        """Call Gemini API for layout detection with retries."""
        import time
        client = _genai.Client(api_key=self._google_key)
        for attempt in range(3):
            try:
                response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt,
                )
                return response.text or ""
            except Exception as e:
                err_str = str(e)
                if "503" in err_str or "429" in err_str or "UNAVAILABLE" in err_str:
                    wait = (attempt + 1) * 5
                    logger.warning("Gemini %s — retrying in %ds", err_str[:60], wait)
                    time.sleep(wait)
                    continue
                logger.error("Gemini failed: %s", e)
                return None
        return None
