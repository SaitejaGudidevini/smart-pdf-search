"""Semantic row enrichment and auto schema generation for Excel RAG.

Converts raw DataFrames into:
1. Semantic row strings — embeddable text with full header context
2. Schema descriptions — rich column metadata for LLM SQL prompts
"""

from __future__ import annotations

import pandas as pd


class ExcelEnricher:
    """Enrich Excel data for RAG: semantic rows + schema descriptions."""

    def generate_semantic_rows(
        self,
        df: pd.DataFrame,
        sheet_name: str,
        formulas: list[str] | None = None,
        header_row_offset: int = 0,
    ) -> list[dict]:
        """Convert each DataFrame row into a semantic string with full context.

        Returns list of dicts:
            {"text": "...", "row_index": 0, "sheet_name": "...", "metadata": {...}}
        """
        formula_map = self._build_formula_map(
            formulas or [], list(df.columns), header_row_offset
        )
        rows = []

        for idx, row in df.iterrows():
            parts = []
            row_formulas = []

            for col in df.columns:
                if col == "__section__":
                    continue  # handled separately above
                val = row[col]
                if pd.notna(val):
                    parts.append(f"{col}: {val}")
                    # Check if this cell has a formula
                    cell_formulas = [
                        f for f in formula_map
                        if f.get("column_name") == col
                        and f.get("data_row") == idx
                    ]
                    for cf in cell_formulas:
                        row_formulas.append(f"{col}={cf['formula']}")

            # Build semantic string with section context if available
            section = None
            if "__section__" in df.columns and pd.notna(row.get("__section__")):
                section = str(row["__section__"])

            if section:
                text = f"Sheet: {sheet_name} | Section: {section} | Row {idx + 1} | {' | '.join(parts)}"
            else:
                text = f"Sheet: {sheet_name} | Row {idx + 1} | {' | '.join(parts)}"

            if row_formulas:
                text += f" | Formulas: {', '.join(row_formulas)}"

            rows.append({
                "text": text,
                "row_index": idx,
                "sheet_name": sheet_name,
                "metadata": {
                    "sheet_name": sheet_name,
                    "row_index": idx,
                    "section": section,
                    "has_formulas": len(row_formulas) > 0,
                },
            })

        return rows

    def generate_schema_description(
        self,
        df: pd.DataFrame,
        table_name: str,
    ) -> str:
        """Build a rich schema description from the DataFrame itself.

        This goes into the LLM system prompt so it can write accurate SQL.
        """
        lines = [
            f"Table: {table_name}",
            f"Rows: {len(df)}",
            f"Columns ({len(df.columns)}):",
        ]

        for col in df.columns:
            if col == "__section__":
                continue  # internal metadata, not a data column
            dtype = str(df[col].dtype)
            non_null = int(df[col].notna().sum())
            unique = df[col].dropna().unique()
            num_unique = len(unique)

            # Sample values (up to 5)
            sample = unique[:5].tolist()
            # Convert numpy types to native Python for clean display
            sample = [
                str(v) if hasattr(v, "isoformat") else v
                for v in sample
            ]

            desc = f"  - {col} ({dtype}, {non_null}/{len(df)} non-null, {num_unique} unique)"

            if num_unique <= 10:
                desc += f" — values: {sample}"
            else:
                desc += f" — e.g.: {sample}"

            # Range for numeric columns
            if df[col].dtype in ("int64", "float64", "int32", "float32"):
                col_min = df[col].min()
                col_max = df[col].max()
                desc += f" — range: [{col_min}, {col_max}]"

            lines.append(desc)

        return "\n".join(lines)

    def generate_all_schemas(
        self,
        dataframes: dict[str, pd.DataFrame],
    ) -> str:
        """Generate schema descriptions for all sheets combined."""
        parts = []
        for table_name, df in dataframes.items():
            parts.append(self.generate_schema_description(df, table_name))
        return "\n\n".join(parts)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_formula_map(
        self,
        formulas: list[str],
        column_names: list[str] | None = None,
        header_row_offset: int = 0,
    ) -> list[dict]:
        """Parse formula strings like 'C5: =SUM(B2:B10)' and map to column names.

        Excel letters (A, B, C) are converted to DataFrame column names by position.
        Row numbers are adjusted for the header offset so they match DataFrame indices.
        """
        import re
        result = []
        for f in formulas:
            match = re.match(r"([A-Z]+)(\d+):\s*(.+)", f)
            if match:
                col_letter, row_num, formula = match.groups()
                col_idx = self._excel_col_to_index(col_letter)
                excel_row = int(row_num)
                # Map Excel row → DataFrame row (subtract header rows + 1 for 0-index)
                data_row = excel_row - header_row_offset - 2  # -1 for header, -1 for 0-index

                col_name = col_letter  # fallback
                if column_names and col_idx < len(column_names):
                    col_name = column_names[col_idx]

                result.append({
                    "column_name": col_name,
                    "data_row": data_row,
                    "formula": formula,
                })
        return result

    @staticmethod
    def _excel_col_to_index(col_letter: str) -> int:
        """Convert Excel column letter to 0-based index. A=0, B=1, Z=25, AA=26."""
        result = 0
        for c in col_letter.upper():
            result = result * 26 + (ord(c) - ord("A") + 1)
        return result - 1
