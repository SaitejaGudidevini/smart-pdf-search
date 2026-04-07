"""Semantic row enrichment and auto schema generation for Excel RAG.

Converts raw DataFrames into:
1. Column classifications — LLM-powered semantic role tagging per column
2. Sheet summaries — top-level metadata (type, entity, metrics, dimensions)
3. Semantic row strings — role-tagged embeddable text with full header context
4. Schema descriptions — rich column metadata for LLM SQL prompts

Inspired by GEDA's "classify first, then process" approach:
  https://github.com/thomasruegg/generative-excel-data-assistant
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import pandas as pd

from cell_dna import SheetDNA, ColumnDNA, format_value_with_context, CellDNA

logger = logging.getLogger(__name__)


# ======================================================================
# Column classification types
# ======================================================================

# Semantic roles a column can play
COLUMN_ROLES = {
    "identifier": "Unique entity name/ID (e.g. company name, employee ID, product SKU)",
    "metric": "Numeric measure that can be aggregated (e.g. revenue, quantity, price)",
    "temporal": "Date, time, year, quarter, or month column",
    "categorical": "Grouping/segmentation dimension (e.g. region, department, status)",
    "descriptive": "Free-text or label that describes an entity (e.g. notes, address)",
    "computed": "Derived/formula column (e.g. profit margin, YoY growth)",
    "ordinal": "Ordered category (e.g. priority: Low/Medium/High, rating: 1-5)",
}


@dataclass
class ColumnClassification:
    """Classification result for a single column."""
    column_name: str
    role: str  # one of COLUMN_ROLES keys
    description: str  # LLM-generated one-line description
    format_hint: str = ""  # e.g. "currency_usd", "percentage", "date_yyyy_mm_dd"

    def to_dict(self) -> dict:
        d = {"column_name": self.column_name, "role": self.role, "description": self.description}
        if self.format_hint:
            d["format_hint"] = self.format_hint
        return d


@dataclass
class SheetSummary:
    """Top-level metadata summary for a single sheet."""
    sheet_name: str
    data_type: str  # e.g. "financial_report", "inventory", "sales_data"
    entity_description: str  # what each row represents
    entity_column: str | None  # primary identifier column
    metrics: list[str] = field(default_factory=list)
    dimensions: list[str] = field(default_factory=list)
    temporal_columns: list[str] = field(default_factory=list)
    row_count: int = 0
    time_range: str = ""  # e.g. "Q1 2023 - Q4 2024"

    def to_dict(self) -> dict:
        return {
            "sheet_name": self.sheet_name,
            "data_type": self.data_type,
            "entity_description": self.entity_description,
            "entity_column": self.entity_column,
            "metrics": self.metrics,
            "dimensions": self.dimensions,
            "temporal_columns": self.temporal_columns,
            "row_count": self.row_count,
            "time_range": self.time_range,
        }


# ======================================================================
# LLM caller (reuses the provider chain from excel_agent.py)
# ======================================================================

class _LLMCaller:
    """Minimal LLM caller with provider fallback chain."""

    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    def call(self, system_prompt: str, user_prompt: str) -> str | None:
        providers = [
            ("Groq", self._call_groq),
            ("Ollama", self._call_ollama),
            ("Anthropic", self._call_anthropic),
            ("OpenAI", self._call_openai),
        ]
        for name, fn in providers:
            try:
                text = fn(system_prompt, user_prompt)
                if text:
                    return text
            except Exception:
                logger.debug("LLM provider %s failed for classification", name, exc_info=True)
                continue
        return None

    def _call_groq(self, system: str, user: str) -> str | None:
        key = os.environ.get("GROQ_API_KEY")
        if not key:
            return None
        with httpx.Client(timeout=self.timeout) as c:
            r = c.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                    "max_tokens": 1000, "temperature": 0,
                },
            )
            if r.status_code != 200:
                return None
            return r.json()["choices"][0]["message"]["content"]

    def _call_ollama(self, system: str, user: str) -> str | None:
        with httpx.Client(timeout=60) as c:
            r = c.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": "llama3.2",
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                    "stream": False,
                },
            )
            if r.status_code != 200:
                return None
            return r.json().get("message", {}).get("content")

    def _call_anthropic(self, system: str, user: str) -> str | None:
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            return None
        with httpx.Client(timeout=self.timeout) as c:
            r = c.post(
                "https://api.anthropic.com/v1/messages",
                headers={"x-api-key": key, "anthropic-version": "2023-06-01", "content-type": "application/json"},
                json={
                    "model": "claude-haiku-4-5-20251001", "max_tokens": 1000,
                    "system": system,
                    "messages": [{"role": "user", "content": user}],
                },
            )
            if r.status_code != 200:
                return None
            return r.json().get("content", [{}])[0].get("text")

    def _call_openai(self, system: str, user: str) -> str | None:
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            return None
        with httpx.Client(timeout=self.timeout) as c:
            r = c.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                    "max_tokens": 1000, "temperature": 0,
                },
            )
            if r.status_code != 200:
                return None
            return r.json()["choices"][0]["message"]["content"]


# ======================================================================
# Column Classifier
# ======================================================================

_CLASSIFY_SYSTEM_PROMPT = """\
You are a data analyst. Classify spreadsheet columns by their semantic role.

Available roles:
- identifier: Unique entity name or ID (company name, employee ID, SKU)
- metric: Numeric measure that can be aggregated (revenue, quantity, price, cost)
- temporal: Date, time, year, quarter, or month
- categorical: Grouping/segmentation dimension (region, department, category, status)
- descriptive: Free-text or label (notes, address, description)
- computed: Derived/formula column (margin %, YoY growth, calculated fields)
- ordinal: Ordered category (priority levels, ratings, grades)

Return ONLY a JSON array. No markdown fences, no explanation."""

_SUMMARY_SYSTEM_PROMPT = """\
You are a data analyst. Given column metadata and sample data from a spreadsheet, \
generate a structured summary of what this sheet contains.

Return ONLY a JSON object with these keys:
- "data_type": string — category like "financial_report", "sales_data", "inventory", \
"employee_records", "invoice", "budget", "time_series", "survey_results", etc.
- "entity_description": string — what each row represents (e.g. "one sales transaction", \
"a monthly revenue entry per region")
- "entity_column": string or null — the primary identifier column name
- "time_range": string — detected time range or "" if none

No markdown fences, no explanation. ONLY the JSON object."""


class ColumnClassifier:
    """LLM-powered column semantic role classifier."""

    def __init__(self):
        self._llm = _LLMCaller()

    def classify_columns(
        self,
        df: pd.DataFrame,
        sheet_name: str,
        sheet_dna: SheetDNA | None = None,
    ) -> list[ColumnClassification]:
        """Classify each column's semantic role using DNA + LLM + heuristic fallback.

        Priority chain:
        1. CellDNA — if openpyxl format metadata gives a confident type, use it
           (e.g. number_format='$#,##0.00' → currency_usd → role=metric)
        2. LLM — for columns where DNA is ambiguous (general/text/number)
        3. Heuristic — dtype-based fallback when no LLM is available

        DNA-resolved columns skip the LLM entirely, saving cost and latency.
        """
        # --- Phase 1: Resolve what we can from DNA alone ---
        dna_resolved: dict[str, ColumnClassification] = {}
        dna_ambiguous: list[str] = []

        if sheet_dna and sheet_dna.column_dna:
            for col in df.columns:
                if col == "__section__":
                    continue
                col_dna = self._find_column_dna(col, sheet_dna)
                if col_dna and col_dna.inferred_role and col_dna.type_confidence >= 0.6:
                    dna_resolved[col] = ColumnClassification(
                        column_name=col,
                        role=col_dna.inferred_role,
                        description=self._dna_description(col, col_dna),
                        format_hint=col_dna.format_hint,
                    )
                    logger.debug(
                        "DNA resolved '%s' → role=%s, type=%s (confidence=%.0f%%)",
                        col, col_dna.inferred_role, col_dna.semantic_type,
                        col_dna.type_confidence * 100,
                    )
                else:
                    dna_ambiguous.append(col)
        else:
            dna_ambiguous = [c for c in df.columns if c != "__section__"]

        if dna_resolved:
            logger.info(
                "DNA resolved %d/%d columns for '%s' — %d need LLM",
                len(dna_resolved), len(dna_resolved) + len(dna_ambiguous),
                sheet_name, len(dna_ambiguous),
            )

        # --- Phase 2: LLM for ambiguous columns ---
        llm_resolved: dict[str, ColumnClassification] = {}
        if dna_ambiguous:
            columns_info = self._build_column_info(df, only_columns=dna_ambiguous)
            if columns_info:
                llm_result = self._classify_via_llm(columns_info, sheet_name)
                if llm_result:
                    for c in llm_result:
                        llm_resolved[c.column_name] = c

        # --- Phase 3: Heuristic fallback for anything still unresolved ---
        results = []
        for col in df.columns:
            if col == "__section__":
                continue
            if col in dna_resolved:
                results.append(dna_resolved[col])
            elif col in llm_resolved:
                results.append(llm_resolved[col])
            else:
                role, desc, fmt = self._infer_column_role(df, col)
                results.append(ColumnClassification(
                    column_name=col, role=role, description=desc, format_hint=fmt,
                ))

        return results

    def _find_column_dna(self, col_name: str, sheet_dna: SheetDNA) -> ColumnDNA | None:
        """Match a DataFrame column name to its SheetDNA ColumnDNA.

        DataFrame columns are cleaned (lowercased, underscored) but DNA uses
        original Excel names. We try exact match first, then fuzzy.
        """
        # Exact match
        if col_name in sheet_dna.column_dna:
            return sheet_dna.column_dna[col_name]

        # Cleaned-name match
        clean = re.sub(r"[^a-z0-9]+", "_", col_name.lower()).strip("_")
        for dna_name, dna in sheet_dna.column_dna.items():
            dna_clean = re.sub(r"[^a-z0-9]+", "_", dna_name.lower()).strip("_")
            if clean == dna_clean:
                return dna

        return None

    def _dna_description(self, col_name: str, col_dna: ColumnDNA) -> str:
        """Build a human-readable description from DNA metadata."""
        parts = [f"{col_dna.semantic_type} column"]
        if col_dna.number_format != "General":
            parts.append(f"format: {col_dna.number_format}")
        if col_dna.has_total_row:
            parts.append("has totals")
        if col_dna.has_formulas:
            parts.append("has formulas")
        if col_dna.has_validation and col_dna.validation_values:
            parts.append(f"allowed: {col_dna.validation_values[:5]}")
        if col_dna.indent_levels:
            parts.append("hierarchical")
        return f"{col_name}: {', '.join(parts)}"

    def generate_sheet_summary(
        self,
        df: pd.DataFrame,
        sheet_name: str,
        classifications: list[ColumnClassification],
    ) -> SheetSummary:
        """Generate a top-level sheet summary using LLM + classifications.

        Falls back to building the summary from classifications alone if LLM
        is unavailable.
        """
        metrics = [c.column_name for c in classifications if c.role == "metric"]
        dimensions = [c.column_name for c in classifications if c.role == "categorical"]
        temporal = [c.column_name for c in classifications if c.role == "temporal"]
        identifiers = [c.column_name for c in classifications if c.role == "identifier"]
        entity_col = identifiers[0] if identifiers else None

        # Detect time range from temporal columns
        time_range = self._detect_time_range(df, temporal)

        # Try LLM for richer summary
        llm_summary = self._summarize_via_llm(df, sheet_name, classifications)
        if llm_summary:
            return SheetSummary(
                sheet_name=sheet_name,
                data_type=llm_summary.get("data_type", "unknown"),
                entity_description=llm_summary.get("entity_description", "one row per record"),
                entity_column=llm_summary.get("entity_column") or entity_col,
                metrics=metrics,
                dimensions=dimensions,
                temporal_columns=temporal,
                row_count=len(df),
                time_range=llm_summary.get("time_range") or time_range,
            )

        # Fallback: build from classifications
        logger.info("LLM unavailable — building summary from heuristics for '%s'", sheet_name)
        data_type = self._infer_data_type(sheet_name, classifications)
        return SheetSummary(
            sheet_name=sheet_name,
            data_type=data_type,
            entity_description="one row per record",
            entity_column=entity_col,
            metrics=metrics,
            dimensions=dimensions,
            temporal_columns=temporal,
            row_count=len(df),
            time_range=time_range,
        )

    # ------------------------------------------------------------------
    # LLM classification
    # ------------------------------------------------------------------

    def _build_column_info(self, df: pd.DataFrame, only_columns: list[str] | None = None) -> list[dict]:
        """Build a compact column profile for the LLM prompt."""
        info = []
        target_cols = only_columns if only_columns is not None else list(df.columns)
        for col in target_cols:
            if col == "__section__" or col not in df.columns:
                continue
            dtype = str(df[col].dtype)
            non_null = int(df[col].notna().sum())
            unique = df[col].dropna().unique()
            sample = unique[:5].tolist()
            sample = [str(v) if hasattr(v, "isoformat") else v for v in sample]

            entry: dict[str, Any] = {
                "name": col,
                "dtype": dtype,
                "non_null": non_null,
                "total": len(df),
                "unique_count": len(unique),
                "samples": sample,
            }
            if df[col].dtype.kind in ("i", "f"):
                entry["min"] = _to_native(df[col].min())
                entry["max"] = _to_native(df[col].max())
            info.append(entry)
        return info

    def _classify_via_llm(
        self, columns_info: list[dict], sheet_name: str
    ) -> list[ColumnClassification] | None:
        user_prompt = (
            f"Sheet name: \"{sheet_name}\"\n\n"
            f"Columns:\n{json.dumps(columns_info, indent=2, default=str)}\n\n"
            "For each column, return a JSON array of objects with keys:\n"
            '  "column_name", "role", "description", "format_hint"\n\n'
            "format_hint examples: currency_usd, currency_eur, percentage, "
            "date_yyyy_mm_dd, integer_count, float_ratio, text, enum\n\n"
            "Return ONLY the JSON array."
        )

        raw = self._llm.call(_CLASSIFY_SYSTEM_PROMPT, user_prompt)
        if not raw:
            return None

        parsed = _parse_json(raw)
        if not isinstance(parsed, list):
            logger.warning("LLM classification returned non-list: %s", raw[:200])
            return None

        results = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            role = item.get("role", "descriptive")
            if role not in COLUMN_ROLES:
                role = "descriptive"
            results.append(ColumnClassification(
                column_name=item.get("column_name", ""),
                role=role,
                description=item.get("description", ""),
                format_hint=item.get("format_hint", ""),
            ))
        return results if results else None

    def _summarize_via_llm(
        self,
        df: pd.DataFrame,
        sheet_name: str,
        classifications: list[ColumnClassification],
    ) -> dict | None:
        class_summary = [c.to_dict() for c in classifications]

        # Include a few sample rows for context
        sample_rows = df.head(3).to_dict(orient="records")
        sample_str = json.dumps(sample_rows, indent=2, default=str)[:2000]

        user_prompt = (
            f"Sheet name: \"{sheet_name}\"\n"
            f"Row count: {len(df)}\n\n"
            f"Column classifications:\n{json.dumps(class_summary, indent=2)}\n\n"
            f"Sample rows (first 3):\n{sample_str}\n\n"
            "Return the JSON summary object."
        )

        raw = self._llm.call(_SUMMARY_SYSTEM_PROMPT, user_prompt)
        if not raw:
            return None

        parsed = _parse_json(raw)
        return parsed if isinstance(parsed, dict) else None

    # ------------------------------------------------------------------
    # Heuristic fallback
    # ------------------------------------------------------------------

    def _classify_heuristic(self, df: pd.DataFrame) -> list[ColumnClassification]:
        """Rule-based classification when no LLM is available."""
        results = []
        for col in df.columns:
            if col == "__section__":
                continue
            role, desc, fmt = self._infer_column_role(df, col)
            results.append(ColumnClassification(
                column_name=col, role=role, description=desc, format_hint=fmt,
            ))
        return results

    def _infer_column_role(self, df: pd.DataFrame, col: str) -> tuple[str, str, str]:
        """Infer a single column's role from dtype + name patterns."""
        name_lower = col.lower().strip()
        dtype = df[col].dtype
        unique_count = df[col].nunique()
        total = len(df)

        # Temporal patterns
        temporal_patterns = r"(date|time|year|month|quarter|week|day|period|timestamp)"
        if re.search(temporal_patterns, name_lower):
            return "temporal", f"Time/date column: {col}", "date"

        # Check if dtype is datetime
        if pd.api.types.is_datetime64_any_dtype(dtype):
            return "temporal", f"Datetime column: {col}", "datetime"

        # Identifier patterns
        id_patterns = r"(^id$|_id$|^name$|^code$|^sku$|^key$|^number$|^no$|^num$|employee|company|customer|vendor|supplier|product)"
        if re.search(id_patterns, name_lower):
            if dtype == "object" or unique_count > total * 0.5:
                return "identifier", f"Entity identifier: {col}", "text"

        # Metric patterns (numeric with aggregation-friendly names)
        metric_patterns = r"(amount|revenue|sales|cost|price|total|sum|count|quantity|qty|volume|profit|loss|budget|expense|income|balance|fee|rate|wage|salary)"
        if re.search(metric_patterns, name_lower) and dtype.kind in ("i", "f"):
            fmt = "currency_usd" if re.search(r"(\$|dollar|usd|price|cost|revenue|sales|income|expense|profit|loss|fee|wage|salary|budget)", name_lower) else "number"
            return "metric", f"Numeric measure: {col}", fmt

        # Generic numeric — if high cardinality, likely metric
        if dtype.kind in ("i", "f"):
            if unique_count > 20 or unique_count > total * 0.3:
                return "metric", f"Numeric column: {col}", "number"
            return "ordinal", f"Numeric category: {col}", "integer_count"

        # Categorical: low-cardinality strings
        if dtype == "object" and unique_count <= 20:
            return "categorical", f"Category column: {col}", "enum"

        # High cardinality string — likely identifier or descriptive
        if dtype == "object":
            if unique_count > total * 0.8:
                return "identifier", f"Unique text: {col}", "text"
            return "descriptive", f"Text column: {col}", "text"

        return "descriptive", f"Column: {col}", ""

    def _detect_time_range(self, df: pd.DataFrame, temporal_cols: list[str]) -> str:
        """Extract time range string from temporal columns."""
        for col in temporal_cols:
            if col not in df.columns:
                continue
            series = df[col].dropna()
            if series.empty:
                continue
            try:
                if pd.api.types.is_datetime64_any_dtype(series):
                    return f"{series.min().strftime('%Y-%m-%d')} to {series.max().strftime('%Y-%m-%d')}"
                if series.dtype.kind in ("i", "f"):
                    mn, mx = int(series.min()), int(series.max())
                    if 1900 <= mn <= 2100:
                        return f"{mn} to {mx}" if mn != mx else str(mn)
            except Exception:
                continue
        return ""

    def _infer_data_type(self, sheet_name: str, classifications: list[ColumnClassification]) -> str:
        """Infer sheet data type from name and column roles."""
        name_lower = sheet_name.lower()
        roles = {c.role for c in classifications}
        hints = " ".join(c.column_name.lower() for c in classifications)

        if re.search(r"(sales|revenue|income)", name_lower + hints):
            return "sales_data"
        if re.search(r"(invoice|billing|payment)", name_lower + hints):
            return "invoice"
        if re.search(r"(inventory|stock|warehouse)", name_lower + hints):
            return "inventory"
        if re.search(r"(employee|staff|hr|personnel)", name_lower + hints):
            return "employee_records"
        if re.search(r"(budget|forecast|plan)", name_lower + hints):
            return "budget"
        if "temporal" in roles and "metric" in roles:
            return "time_series"
        if "metric" in roles:
            return "financial_report"
        return "tabular_data"


# ======================================================================
# ExcelEnricher (updated with classification support)
# ======================================================================

class ExcelEnricher:
    """Enrich Excel data for RAG: classify columns, then generate semantic rows + schemas."""

    def __init__(self):
        self._classifier = ColumnClassifier()
        # Cache: sheet_name -> classifications
        self._classifications: dict[str, list[ColumnClassification]] = {}
        # Cache: sheet_name -> summary
        self._summaries: dict[str, SheetSummary] = {}
        # CellDNA from workbook (set by classify_and_summarize)
        self._workbook_dna: dict[str, SheetDNA] = {}
        # Stage 1 workbook extraction context (set by caller when available)
        self._stage1_result: Any | None = None
        self._stage1_tables_by_output_name: dict[str, Any] = {}

    def set_stage1_result(self, stage1_result: Any | None) -> None:
        """Attach Stage 1 extraction output so semantic rows can use provenance."""
        self._stage1_result = stage1_result
        self._stage1_tables_by_output_name = {}
        if not stage1_result or not getattr(stage1_result, "tables", None):
            return

        seen_names: dict[str, int] = {}
        for table in stage1_result.tables:
            dataframe = getattr(table, "dataframe", None)
            if dataframe is None or dataframe.empty:
                continue

            clean_name = self._clean_name(table.sheet_name)
            count = seen_names.get(clean_name, 0)
            seen_names[clean_name] = count + 1
            if count:
                clean_name = f"{clean_name}_{count}"
            self._stage1_tables_by_output_name[clean_name] = table

    def classify_and_summarize(
        self,
        dataframes: dict[str, pd.DataFrame],
        workbook_dna: dict[str, SheetDNA] | None = None,
    ) -> dict[str, dict]:
        """Run DNA + LLM classification + summary for all sheets.

        Call before generate_semantic_rows.

        Args:
            dataframes: {sheet_name: DataFrame}
            workbook_dna: {original_sheet_name: SheetDNA} from CellDNA extraction.
                When provided, columns with confident format metadata skip LLM.

        Returns a dict keyed by sheet name with "classifications", "summary",
        and "dna_resolved_count".
        """
        # Store DNA for use in semantic row generation
        self._workbook_dna = workbook_dna or {}

        result = {}
        for sheet_name, df in dataframes.items():
            # Find matching SheetDNA (sheet names may be cleaned)
            sheet_dna = self._match_sheet_dna(sheet_name, workbook_dna)

            classifications = self._classifier.classify_columns(df, sheet_name, sheet_dna)
            self._classifications[sheet_name] = classifications

            summary = self._classifier.generate_sheet_summary(df, sheet_name, classifications)
            self._summaries[sheet_name] = summary

            dna_count = sum(1 for c in classifications
                           if sheet_dna and self._classifier._find_column_dna(c.column_name, sheet_dna))

            result[sheet_name] = {
                "classifications": [c.to_dict() for c in classifications],
                "summary": summary.to_dict(),
                "dna_resolved_count": dna_count,
            }
            logger.info(
                "Classified '%s': type=%s, %d metrics, %d dimensions, entity=%s (DNA resolved %d cols)",
                sheet_name, summary.data_type, len(summary.metrics),
                len(summary.dimensions), summary.entity_column, dna_count,
            )
        return result

    def _match_sheet_dna(
        self, clean_name: str, workbook_dna: dict[str, SheetDNA] | None,
    ) -> SheetDNA | None:
        """Match a cleaned sheet name to its SheetDNA using fuzzy matching.

        Handles the case where Gemini/LlamaParse renames sheets (e.g. to "sheet")
        but the original Excel sheet name is "Financial Summary".
        """
        if not workbook_dna:
            return None

        # Exact match
        if clean_name in workbook_dna:
            return workbook_dna[clean_name]

        # Cleaned-name match
        target = re.sub(r"[^a-z0-9]+", "_", clean_name.lower()).strip("_")
        for orig_name, dna in workbook_dna.items():
            cleaned = re.sub(r"[^a-z0-9]+", "_", orig_name.lower()).strip("_")
            if target == cleaned:
                return dna

        # Substring match (e.g. "annual_sales" matches "Annual Sales Report 2026")
        for orig_name, dna in workbook_dna.items():
            cleaned = re.sub(r"[^a-z0-9]+", "_", orig_name.lower()).strip("_")
            if target in cleaned or cleaned in target:
                return dna

        # If only one sheet in DNA, assume it's the match
        # (common when Gemini renames sheets to generic names)
        if len(workbook_dna) == 1:
            return next(iter(workbook_dna.values()))

        return None

    def generate_semantic_rows(
        self,
        df: pd.DataFrame,
        sheet_name: str,
        formulas: list[str] | None = None,
        header_row_offset: int = 0,
    ) -> list[dict]:
        """Convert each DataFrame row into a role-tagged semantic string.

        If classify_and_summarize() was called first, uses role tags:
            [Entity: Acme Corp] [Metric: Revenue = $50,000] [Dimension: Region = West]

        When CellDNA is available, values are formatted with their true format:
            [Metric: Revenue = $1,234,567.89 (USD)] instead of [Metric: Revenue = 1234567.89]

        Otherwise falls back to the original flat format for backward compatibility.
        """
        formula_map = self._build_formula_map(
            formulas or [], list(df.columns), header_row_offset
        )

        # Build role lookup from cached classifications
        role_map: dict[str, ColumnClassification] = {}
        if sheet_name in self._classifications:
            for c in self._classifications[sheet_name]:
                role_map[c.column_name] = c

        # Build DNA-based format lookup for value rendering
        sheet_dna = self._match_sheet_dna(sheet_name, self._workbook_dna)
        col_dna_map: dict[str, ColumnDNA] = {}
        if sheet_dna:
            for col in df.columns:
                if col == "__section__":
                    continue
                cdna = self._classifier._find_column_dna(col, sheet_dna) if hasattr(self, '_classifier') else None
                if cdna:
                    col_dna_map[col] = cdna

        # Get summary context for richer prefix
        summary = self._summaries.get(sheet_name)
        stage1_table = self._stage1_tables_by_output_name.get(sheet_name)
        stage1_cells = self._get_stage1_cells_for_sheet(stage1_table.sheet_name) if stage1_table else {}

        rows = []
        for idx, row in df.iterrows():
            parts = []
            row_formulas = []
            stage1_context = self._get_stage1_row_context(stage1_table, idx, stage1_cells)
            use_stage1_primary = bool(stage1_context.get("semantic_fragments"))

            if not use_stage1_primary:
                for col in df.columns:
                    if col == "__section__":
                        continue
                    val = row[col]
                    if pd.notna(val):
                        # Format value using DNA if available
                        display_val = self._format_cell_value(val, col, col_dna_map)

                        if col in role_map:
                            tag = _role_tag(role_map[col].role)
                            parts.append(f"[{tag}: {col} = {display_val}]")
                        else:
                            parts.append(f"{col}: {display_val}")

                        cell_formulas = [
                            f for f in formula_map
                            if f.get("column_name") == col
                            and f.get("data_row") == idx
                        ]
                        for cf in cell_formulas:
                            row_formulas.append(f"{col}={cf['formula']}")

            # Build prefix with section and summary context
            section = None
            if not use_stage1_primary and "__section__" in df.columns and pd.notna(row.get("__section__")):
                section = str(row["__section__"])
            if not section and stage1_context.get("section_path"):
                section = " > ".join(stage1_context["section_path"])

            prefix_parts = [f"Sheet: {sheet_name}"]
            if summary:
                prefix_parts.append(f"Type: {summary.data_type}")
            if section:
                prefix_parts.append(f"Section: {section}")
            if stage1_context.get("column_paths"):
                prefix_parts.append(
                    "Columns: " + " || ".join(stage1_context["column_paths"])
                )
            if stage1_context.get("units"):
                prefix_parts.append(f"Units: {stage1_context['units']}")
            prefix_parts.append(f"Row {idx + 1}")

            if use_stage1_primary:
                text = f"{' | '.join(prefix_parts)} | Facts: " + " || ".join(stage1_context["semantic_fragments"])
            else:
                text = f"{' | '.join(prefix_parts)} | {' '.join(parts)}"

            if row_formulas:
                text += f" | Formulas: {', '.join(row_formulas)}"
            elif stage1_context.get("semantic_fragments") and not use_stage1_primary:
                text += " | Provenance: " + " || ".join(stage1_context["semantic_fragments"])

            metadata: dict[str, Any] = {
                "sheet_name": sheet_name,
                "row_index": idx,
                "section": section,
                "has_formulas": len(row_formulas) > 0,
            }
            if summary:
                metadata["data_type"] = summary.data_type
                metadata["entity_column"] = summary.entity_column
            if stage1_context:
                if stage1_table:
                    metadata["original_sheet_name"] = stage1_table.sheet_name
                metadata["source_cells"] = stage1_context.get("source_cells", {})
                metadata["source_labels"] = stage1_context.get("source_labels", {})
                metadata["section_path"] = stage1_context.get("section_path", [])
                metadata["column_paths"] = stage1_context.get("column_paths", [])
                metadata["units"] = stage1_context.get("units")
                metadata["row_semantic_texts"] = stage1_context.get("semantic_fragments", [])

            rows.append({
                "text": text,
                "row_index": idx,
                "sheet_name": sheet_name,
                "metadata": metadata,
            })

        return rows

    def _format_cell_value(
        self, value: Any, col_name: str, col_dna_map: dict[str, ColumnDNA],
    ) -> str:
        """Format a cell value using DNA metadata when available.

        Turns raw 1234567.89 into $1,234,567.89 when DNA says it's currency.
        Turns raw 0.15 into 15.0% when DNA says it's percentage.
        """
        cdna = col_dna_map.get(col_name)
        if not cdna:
            return str(value)

        st = cdna.semantic_type

        if st.startswith("currency"):
            symbol = {"currency_usd": "$", "currency_eur": "€", "currency_gbp": "£",
                      "currency_jpy": "¥", "currency_inr": "₹"}.get(st, "$")
            try:
                return f"{symbol}{float(value):,.2f}"
            except (ValueError, TypeError):
                return str(value)

        if st == "accounting":
            try:
                num = float(value)
                return f"{num:,.2f}" if num >= 0 else f"({abs(num):,.2f})"
            except (ValueError, TypeError):
                return str(value)

        if st == "percentage":
            try:
                num = float(value)
                if abs(num) <= 1.0:
                    return f"{num * 100:.1f}%"
                return f"{num:.1f}%"
            except (ValueError, TypeError):
                return str(value)

        if st in ("date", "datetime"):
            if hasattr(value, "strftime"):
                return value.strftime("%Y-%m-%d")
            return str(value)

        return str(value)

    def _get_stage1_cells_for_sheet(self, original_sheet_name: str) -> dict[str, dict]:
        """Index Stage 1 cells by address for a sheet."""
        if not self._stage1_result:
            return {}
        cells_by_sheet = getattr(self._stage1_result, "cells_by_sheet", {}) or {}
        cells = cells_by_sheet.get(original_sheet_name, [])
        return {getattr(cell, "address", ""): cell for cell in cells if getattr(cell, "address", None)}

    def _get_stage1_row_context(
        self,
        stage1_table: Any | None,
        row_index: int,
        stage1_cells: dict[str, Any],
    ) -> dict[str, Any]:
        """Return provenance-aware context for a semantic row."""
        if not stage1_table:
            return {}

        row_provenance = getattr(stage1_table, "row_provenance", []) or []
        if row_index >= len(row_provenance):
            return {}

        row_meta = row_provenance[row_index]
        source_cells = row_meta.get("source_cells", {}) if isinstance(row_meta, dict) else {}
        semantic_fragments: list[str] = []
        section_path: list[str] = []
        column_paths: list[str] = []
        unit_context = None
        source_labels: dict[str, str] = {}

        for column_name, address in source_cells.items():
            cell = stage1_cells.get(address)
            if not cell:
                continue
            semantic_text = getattr(cell, "semantic_text", None)
            if semantic_text:
                semantic_fragments.append(semantic_text)
            cell_section = list(getattr(cell, "section_path", []) or [])
            if cell_section:
                section_path = cell_section
            cell_column_path = list(getattr(cell, "column_header_path", []) or [])
            if cell_column_path:
                joined = " > ".join(cell_column_path)
                if joined not in column_paths:
                    column_paths.append(joined)
            if not unit_context:
                unit_context = getattr(cell, "unit_context", None)
            source_labels[column_name] = getattr(cell, "row_label", None) or column_name

        if not section_path and isinstance(row_meta, dict):
            section_path = list(row_meta.get("section_path", []))

        return {
            "source_cells": source_cells,
            "source_labels": source_labels,
            "section_path": section_path,
            "column_paths": column_paths,
            "units": unit_context,
            "semantic_fragments": semantic_fragments[:8],
        }

    def generate_schema_description(
        self,
        df: pd.DataFrame,
        table_name: str,
    ) -> str:
        """Build a rich schema description from the DataFrame + classifications + DNA.

        This goes into the LLM system prompt so it can write accurate SQL.
        When DNA is available, includes true format info (currency, percentage, etc.)
        """
        lines = [f"Table: {table_name}", f"Rows: {len(df)}"]

        # Add summary context if available
        summary = self._summaries.get(table_name)
        if summary:
            lines.append(f"Data type: {summary.data_type}")
            lines.append(f"Each row represents: {summary.entity_description}")
            if summary.entity_column:
                lines.append(f"Primary identifier: {summary.entity_column}")
            if summary.metrics:
                lines.append(f"Metric columns (aggregatable): {', '.join(summary.metrics)}")
            if summary.dimensions:
                lines.append(f"Dimension columns (group-by): {', '.join(summary.dimensions)}")
            if summary.temporal_columns:
                lines.append(f"Temporal columns: {', '.join(summary.temporal_columns)}")
            if summary.time_range:
                lines.append(f"Time range: {summary.time_range}")

        # Build role lookup
        role_map: dict[str, ColumnClassification] = {}
        if table_name in self._classifications:
            for c in self._classifications[table_name]:
                role_map[c.column_name] = c

        # Build DNA lookup for enriched format info
        sheet_dna = self._match_sheet_dna(table_name, self._workbook_dna)
        col_dna_map: dict[str, ColumnDNA] = {}
        if sheet_dna:
            for col in df.columns:
                if col == "__section__":
                    continue
                cdna = self._classifier._find_column_dna(col, sheet_dna)
                if cdna:
                    col_dna_map[col] = cdna

        lines.append(f"Columns ({len(df.columns)}):")

        for col in df.columns:
            if col == "__section__":
                continue
            dtype = str(df[col].dtype)
            non_null = int(df[col].notna().sum())
            unique = df[col].dropna().unique()
            num_unique = len(unique)

            sample = unique[:5].tolist()
            sample = [str(v) if hasattr(v, "isoformat") else v for v in sample]

            # Role annotation
            role_str = ""
            col_role = role_map.get(col)
            if col_role:
                role_str = f" [role: {col_role.role}]"
                if col_role.format_hint:
                    role_str += f" [format: {col_role.format_hint}]"

            # DNA annotation — adds true Excel format info
            cdna = col_dna_map.get(col)
            if cdna:
                if cdna.number_format != "General":
                    role_str += f" [excel_format: {cdna.number_format}]"
                if cdna.has_total_row:
                    role_str += " [has_totals]"
                if cdna.has_formulas:
                    role_str += " [computed]"
                if cdna.has_validation and cdna.validation_values:
                    role_str += f" [allowed: {cdna.validation_values}]"

            desc = f"  - {col} ({dtype}, {non_null}/{len(df)} non-null, {num_unique} unique){role_str}"

            # For identifier/categorical columns with <=30 unique values,
            # show ALL values so the LLM can write exact WHERE clauses
            is_label_col = col_role and col_role.role in ("identifier", "categorical", "descriptive")
            if is_label_col and num_unique <= 30:
                all_vals = [str(v) for v in unique.tolist()]
                desc += f" -- ALL values: {all_vals}"
            elif num_unique <= 10:
                desc += f" -- values: {sample}"
            else:
                desc += f" -- e.g.: {sample}"

            if df[col].dtype in ("int64", "float64", "int32", "float32"):
                col_min = df[col].min()
                col_max = df[col].max()
                desc += f" -- range: [{col_min}, {col_max}]"

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

    def build_chunks(
        self,
        dataframes: dict[str, pd.DataFrame],
        formulas: dict[str, list[str]],
        filename: str,
        document_key: str,
        schema_description: str,
        mayan_doc_id: int | None = None,
    ) -> list:
        """Convert semantic rows into Chunk objects for embedding.

        Creates one parent chunk per sheet (summary context) and one child
        chunk per row (role-tagged semantic text). Only child chunks get
        embedded; parent chunks provide retrieval context.

        This replaces the old path of extract_structure() → Gemini → markdown
        → ChunkingPipeline for Excel files.
        """
        from chunking_pipeline import Chunk

        chunks: list[Chunk] = []
        file_stem = Path(filename).stem
        sheet_summaries = self.get_all_summaries()

        for sheet_name, df in dataframes.items():
            summary = self._summaries.get(sheet_name)
            parent_id = f"{document_key}:{sheet_name}"

            # --- Parent chunk: sheet summary (not embedded, used for context) ---
            summary_lines = [f"Sheet: {sheet_name} from '{filename}'"]
            if summary:
                summary_lines.append(f"Data type: {summary.data_type}")
                summary_lines.append(f"Each row represents: {summary.entity_description}")
                summary_lines.append(f"Rows: {summary.row_count}")
                if summary.entity_column:
                    summary_lines.append(f"Primary identifier: {summary.entity_column}")
                if summary.metrics:
                    summary_lines.append(f"Metrics: {', '.join(summary.metrics)}")
                if summary.dimensions:
                    summary_lines.append(f"Dimensions: {', '.join(summary.dimensions)}")
                if summary.temporal_columns:
                    summary_lines.append(f"Temporal: {', '.join(summary.temporal_columns)}")
                if summary.time_range:
                    summary_lines.append(f"Time range: {summary.time_range}")

            parent_text = "\n".join(summary_lines)
            parent_meta = {
                "chunk_type": "parent",
                "parent_id": parent_id,
                "section": sheet_name,
                "page_number": 1,
                "doc_type": "spreadsheet",
                "document_name": filename,
                "document_key": document_key,
                "file_type": "spreadsheet",
                "sheet_name": sheet_name,
                "schema_description": schema_description,
                "sheet_summaries": sheet_summaries,
            }
            if mayan_doc_id:
                parent_meta["mayan_doc_id"] = mayan_doc_id

            chunks.append(Chunk(text=parent_text, metadata=parent_meta))

            # --- Child chunks: one per semantic row (these get embedded) ---
            sheet_formulas = formulas.get(sheet_name, [])
            rows = self.generate_semantic_rows(df, sheet_name, sheet_formulas)

            for row in rows:
                # Build enriched text with context prefix
                enriched = (
                    f"From spreadsheet '{file_stem}', sheet '{sheet_name}'. "
                    f"{summary.entity_description if summary else 'One row per record'}.\n\n"
                    f"{row['text']}"
                )

                child_meta = {
                    "chunk_type": "child",
                    "parent_id": parent_id,
                    "section": sheet_name,
                    "page_number": 1,
                    "doc_type": "spreadsheet",
                    "document_name": filename,
                    "document_key": document_key,
                    "file_type": "spreadsheet",
                    "sheet_name": sheet_name,
                    "row_index": row["row_index"],
                    "enriched_text": enriched,
                    "start_line": row["row_index"],
                    "end_line": row["row_index"],
                }
                if mayan_doc_id:
                    child_meta["mayan_doc_id"] = mayan_doc_id
                # Carry forward row-level metadata
                if row.get("metadata"):
                    for k in ("data_type", "entity_column", "has_formulas"):
                        if k in row["metadata"]:
                            child_meta[k] = row["metadata"][k]

                chunks.append(Chunk(text=row["text"], metadata=child_meta))

        return chunks

    def get_all_summaries(self) -> dict[str, dict]:
        """Return cached sheet summaries as JSON-serializable dicts."""
        return {name: s.to_dict() for name, s in self._summaries.items()}

    def get_all_classifications(self) -> dict[str, list[dict]]:
        """Return cached column classifications as JSON-serializable dicts."""
        return {
            name: [c.to_dict() for c in classes]
            for name, classes in self._classifications.items()
        }

    @staticmethod
    def _clean_name(name: str) -> str:
        """Normalize table/sheet names the same way Stage 1 parse outputs do."""
        cleaned = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_")
        return cleaned or "sheet"

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_formula_map(
        self,
        formulas: list[str],
        column_names: list[str] | None = None,
        header_row_offset: int = 0,
    ) -> list[dict]:
        """Parse formula strings like 'C5: =SUM(B2:B10)' and map to column names."""
        result = []
        for f in formulas:
            match = re.match(r"([A-Z]+)(\d+):\s*(.+)", f)
            if match:
                col_letter, row_num, formula = match.groups()
                col_idx = self._excel_col_to_index(col_letter)
                excel_row = int(row_num)
                data_row = excel_row - header_row_offset - 2

                col_name = col_letter
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


# ======================================================================
# Helpers
# ======================================================================

def _role_tag(role: str) -> str:
    """Map role to a display tag for semantic rows."""
    tags = {
        "identifier": "Entity",
        "metric": "Metric",
        "temporal": "Time",
        "categorical": "Dimension",
        "descriptive": "Info",
        "computed": "Computed",
        "ordinal": "Rank",
    }
    return tags.get(role, "Field")


def _parse_json(raw: str) -> Any:
    """Extract and parse JSON from LLM output, handling markdown fences."""
    text = raw.strip()
    # Strip markdown code fences
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON array or object in the text
        for pattern in [r"\[[\s\S]*\]", r"\{[\s\S]*\}"]:
            match = re.search(pattern, text)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    continue
    return None


def _to_native(value: Any) -> Any:
    """Convert numpy/pandas scalar to native Python type."""
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, float) and (value != value):
        return None
    return value
