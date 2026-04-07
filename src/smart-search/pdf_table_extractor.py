"""Extract DataFrames from markdown tables found in PDF chunks.

When a PDF contains tabular data (financial statements, reports, invoices),
this module parses the markdown tables into pandas DataFrames so the SQL
agent can query them — even though the source file is a PDF, not Excel.

Usage:
    extractor = PDFTableExtractor()
    dataframes = extractor.extract_tables(chunks)
    # Returns: {"income_statement": df1, "balance_sheet": df2, ...}
"""

from __future__ import annotations

import io
import re
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


class PDFTableExtractor:
    """Parse markdown tables from PDF chunks into queryable DataFrames."""

    def extract_tables(self, chunks: list) -> dict[str, pd.DataFrame]:
        """Scan chunks for markdown tables and convert to DataFrames.

        Returns a dict of table_name → DataFrame. Only returns tables with
        at least 2 rows and 2 columns (filters out trivial fragments).
        """
        dataframes: dict[str, pd.DataFrame] = {}
        table_idx = 0

        for chunk in chunks:
            text = chunk.text if hasattr(chunk, "text") else str(chunk)
            meta = chunk.metadata if hasattr(chunk, "metadata") else {}
            section = meta.get("section", "")

            # Find markdown tables in the chunk
            tables = self._find_markdown_tables(text)

            for md_table in tables:
                df = self._parse_markdown_table(md_table)
                if df is not None and len(df) >= 1 and len(df.columns) >= 2:
                    # Generate a clean table name from section
                    name = self._table_name(section, table_idx)
                    dataframes[name] = df
                    table_idx += 1
                    logger.info(
                        "Extracted table '%s' from PDF: %d rows x %d cols",
                        name, len(df), len(df.columns),
                    )

        return dataframes

    def _find_markdown_tables(self, text: str) -> list[str]:
        """Find all markdown table blocks in text."""
        tables = []
        lines = text.split("\n")
        current_table: list[str] = []
        in_table = False

        for line in lines:
            stripped = line.strip()
            # Strip [Table] prefix that the chunking pipeline adds
            if stripped.startswith("[Table]"):
                stripped = stripped[len("[Table]"):].strip()
            if stripped.startswith("|") and "|" in stripped[1:]:
                in_table = True
                current_table.append(stripped)
            else:
                if in_table and current_table:
                    if len(current_table) >= 3:
                        tables.append("\n".join(current_table))
                    current_table = []
                    in_table = False

        if current_table and len(current_table) >= 3:
            tables.append("\n".join(current_table))

        return tables

    def _parse_markdown_table(self, md_table: str) -> pd.DataFrame | None:
        """Parse a markdown table string into a DataFrame."""
        lines = md_table.strip().split("\n")
        if len(lines) < 3:
            return None

        # Determine column count from separator row (most reliable)
        sep_idx = None
        num_cols = 0
        for i in range(1, min(4, len(lines))):
            if re.match(r"^\|[\s\-:|]+\|$", lines[i].strip()):
                sep_idx = i
                num_cols = len(lines[i].strip().split("|")) - 2  # exclude leading/trailing empty
                break

        if sep_idx is None or num_cols < 2:
            return None

        # Parse header — use all lines before separator as header context
        header_cells = self._parse_row(lines[0])
        # Pad/trim to num_cols
        header_cells = (header_cells + [f"col_{j}" for j in range(num_cols)])[:num_cols]
        # Replace empty headers
        for j in range(len(header_cells)):
            if not header_cells[j]:
                header_cells[j] = f"col_{j}"

        data_start = sep_idx + 1

        # Parse data rows, skipping sub-header rows (rows where all non-empty
        # cells contain bold markdown or date patterns — not actual data)
        rows = []
        for line in lines[data_start:]:
            row = self._parse_row(line)
            if not row:
                continue
            # Pad or trim to match column count
            padded = (row + [""] * num_cols)[:num_cols]
            # Skip sub-header rows (all cells are bold text or empty)
            non_empty = [c for c in padded if c.strip()]
            if non_empty and all(re.match(r"^\*\*.*\*\*$", c.strip()) for c in non_empty):
                # This is a sub-header row (e.g., "**March 29, 2025**")
                # Merge into column names instead of treating as data
                for j, cell in enumerate(padded):
                    cleaned = re.sub(r"\*\*(.+?)\*\*", r"\1", cell).strip()
                    cleaned = re.sub(r"<br\s*/?>", " ", cleaned).strip()
                    if cleaned and header_cells[j].startswith("col_"):
                        header_cells[j] = self._clean_column_name(cleaned)
                    elif cleaned:
                        header_cells[j] = self._clean_column_name(
                            header_cells[j] + " " + cleaned
                        )
                continue
            rows.append(padded)

        if not rows:
            return None

        # Re-deduplicate headers after sub-header merge
        seen_h: dict[str, int] = {}
        for j in range(len(header_cells)):
            if header_cells[j] in seen_h:
                seen_h[header_cells[j]] += 1
                header_cells[j] = f"{header_cells[j]}_{seen_h[header_cells[j]]}"
            else:
                seen_h[header_cells[j]] = 0

        df = pd.DataFrame(rows, columns=header_cells)

        # Clean up: remove bold markdown, $, commas, strip whitespace
        for col in df.columns:
            df[col] = df[col].apply(self._clean_cell)

        # Deduplicate column names before numeric conversion
        seen: dict[str, int] = {}
        new_cols = []
        for c in df.columns:
            if c in seen:
                seen[c] += 1
                new_cols.append(f"{c}_{seen[c]}")
            else:
                seen[c] = 0
                new_cols.append(c)
        df.columns = new_cols

        # Try to convert numeric columns
        for col in df.columns:
            series = df[col]
            if not isinstance(series, pd.Series):
                continue
            try:
                converted = pd.to_numeric(series, errors="coerce")
                if converted.notna().sum() > 0.5 * series.notna().sum():
                    df[col] = converted
            except Exception:
                pass

        # Clean column names
        df.columns = [self._clean_column_name(c) for c in df.columns]

        # Drop empty rows and columns
        df = df.dropna(how="all")
        df = df.loc[:, df.notna().any()]

        return df if len(df) > 0 else None

    def _parse_row(self, line: str) -> list[str]:
        """Parse a markdown table row into cells, preserving empty cells."""
        line = line.strip()
        if not line.startswith("|"):
            return []
        # Split by | — keep empty cells (important for table structure)
        cells = line.split("|")
        # Remove first and last entries (before leading | and after trailing |)
        if cells and cells[0].strip() == "":
            cells = cells[1:]
        if cells and cells[-1].strip() == "":
            cells = cells[:-1]
        cells = [c.strip() for c in cells]
        # If all cells are just --- separators, skip
        if cells and all(re.match(r"^[-:]+$", c) for c in cells if c):
            return []
        # If all cells are empty, skip (blank spacer row)
        if all(c == "" for c in cells):
            return []
        return cells

    def _clean_cell(self, val: str) -> str:
        """Clean a markdown cell value."""
        if not isinstance(val, str):
            return val
        # Remove bold markers
        val = re.sub(r"\*\*(.+?)\*\*", r"\1", val)
        # Remove <br> tags
        val = re.sub(r"<br\s*/?>", " ", val)
        # Remove $ prefix and commas from numbers
        cleaned = re.sub(r"^\$\s*", "", val.strip())
        cleaned = cleaned.replace(",", "")
        # Remove parentheses notation for negatives: (123) → -123
        paren_match = re.match(r"^\((\d[\d,.]*)\)$", cleaned)
        if paren_match:
            cleaned = f"-{paren_match.group(1)}"
        return cleaned.strip()

    def _clean_column_name(self, name: str) -> str:
        """Clean a column name for SQL compatibility."""
        if not isinstance(name, str):
            name = str(name)
        # Remove bold, <br>, special chars
        name = re.sub(r"\*\*(.+?)\*\*", r"\1", name)
        name = re.sub(r"<br\s*/?>", " ", name)
        name = re.sub(r"[^a-zA-Z0-9_\s]", "", name).strip()
        name = re.sub(r"\s+", "_", name).lower()
        if not name:
            name = "col"
        if name[0].isdigit():
            name = "c_" + name
        return name

    def _table_name(self, section: str, idx: int) -> str:
        """Generate a clean table name from section title."""
        if section:
            clean = re.sub(r"[^a-zA-Z0-9_\s]", "", section).strip()
            clean = re.sub(r"\s+", "_", clean).lower()
            if clean:
                return clean[:50]
        return f"table_{idx}"
