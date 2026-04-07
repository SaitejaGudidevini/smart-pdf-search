"""Excel Storage Manager — SQLite per-document storage with SQL execution.

Stage 3 of the Excel RAG Pipeline (RE-52): Hybrid Indexing.

For each indexed Excel file, creates a SQLite database containing the
parsed DataFrames as queryable tables. A registry in PostgreSQL tracks
all SQLite databases and their schemas so the query router can decide
between semantic search and direct SQL execution.

Usage:
    manager = ExcelStorageManager()
    manager.store_excel("mayan:42", "budget.xlsx", {"Sheet1": df1, "Q2": df2})
    results = manager.execute_sql("mayan:42", "SELECT SUM(amount) FROM sheet1")
"""

from __future__ import annotations

import json
import logging
import os
import re
import sqlite3
from pathlib import Path
from typing import Any

import pandas as pd
import psycopg
from psycopg.rows import dict_row

logger = logging.getLogger(__name__)

# SQL keywords that indicate a computation/aggregation query
_SQL_SIGNAL_WORDS = re.compile(
    r"\b("
    r"sum|total|average|avg|mean|count|how many|how much|"
    r"max|maximum|highest|largest|biggest|most|"
    r"min|minimum|lowest|smallest|least|"
    r"greater|less|more than|fewer|between|"
    r"filter|where|group|sort|order|rank|top|bottom|"
    r"compare|difference|ratio|percent|percentage|growth|change|"
    r"above|below|exceed|over|under|"
    r"calculate|compute|add up|subtract|multiply|divide|"
    r"cost|costs|revenue|income|expense|expenses|profit|loss|"
    r"assets|liabilities|equity|cash|debt|margin|"
    r"balance|operating|net|gross|"
    r"include all|show me|list all|give me|what was|what were"
    r")\b",
    re.IGNORECASE,
)

# Patterns that strongly indicate semantic/text retrieval
_SEMANTIC_SIGNAL_WORDS = re.compile(
    r"\b("
    r"what is|what are|explain|describe|tell me about|"
    r"define|meaning|overview|summary|summarize|"
    r"who|why|how does|how do|"
    r"find information|search for|look up|"
    r"related to|about|regarding|concerning"
    r")\b",
    re.IGNORECASE,
)

_PROVENANCE_SIGNAL_WORDS = re.compile(
    r"\b("
    r"where did|came from|come from|source of|source for|origin of|"
    r"what does this number|what is this value|which cell|which row|which section|"
    r"which year|which period|line item|belongs to|represented by|"
    r"what does .* represent|trace|provenance"
    r")\b",
    re.IGNORECASE,
)

# Statements that mutate data -- must be rejected
_MUTATION_PATTERN = re.compile(
    r"^\s*(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|TRUNCATE|REPLACE|ATTACH|DETACH|PRAGMA|VACUUM)\b",
    re.IGNORECASE,
)


class ExcelStorageManager:
    """Manages per-document SQLite databases for Excel files.

    Each Excel file gets its own SQLite DB with one table per sheet.
    A registry table in PostgreSQL tracks all databases and their schemas.
    """

    def __init__(self):
        self.db_dir = Path(os.environ.get("EXCEL_DB_DIR", "/tmp/excel_dbs"))
        self.db_dir.mkdir(parents=True, exist_ok=True)

        # PostgreSQL connection parameters (same source as storage_pg.py)
        self._pg_host = os.environ.get("RAG_PGHOST", os.environ.get("POSTGRES_HOST", "postgresql"))
        self._pg_port = int(os.environ.get("RAG_PGPORT", os.environ.get("POSTGRES_PORT", "5432")))
        self._pg_dbname = os.environ.get("RAG_PGDATABASE", os.environ.get("POSTGRES_DB", "mayan"))
        self._pg_user = os.environ.get("RAG_PGUSER", os.environ.get("POSTGRES_USER", "mayan"))
        self._pg_password = os.environ.get("RAG_PGPASSWORD", os.environ.get("POSTGRES_PASSWORD", "mayan"))
        self._pg_schema = os.environ.get("RAG_PG_SCHEMA", "rag")

        self._pg_available = False
        try:
            self._ensure_registry_table()
            self._pg_available = True
        except Exception as e:
            print(f"[ExcelStorage] PostgreSQL not available, using SQLite only: {e}")

        # Local registry fallback (when PostgreSQL is not available)
        self._local_registry: dict[str, dict] = {}

    # ------------------------------------------------------------------
    # PostgreSQL connection
    # ------------------------------------------------------------------

    def _pg_connect(self):
        return psycopg.connect(
            host=self._pg_host,
            port=self._pg_port,
            dbname=self._pg_dbname,
            user=self._pg_user,
            password=self._pg_password,
            row_factory=dict_row,
        )

    def _ensure_registry_table(self):
        """Create the rag.excel_databases registry table if it does not exist."""
        with self._pg_connect() as conn:
            conn.execute(f"CREATE SCHEMA IF NOT EXISTS {self._pg_schema}")
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self._pg_schema}.excel_databases (
                    document_key text PRIMARY KEY,
                    document_name text NOT NULL,
                    db_path text NOT NULL,
                    schema_json jsonb NOT NULL,
                    sheet_names jsonb NOT NULL,
                    total_rows integer DEFAULT 0,
                    created_at timestamptz DEFAULT now()
                )
                """
            )

    # ------------------------------------------------------------------
    # Store Excel as SQLite
    # ------------------------------------------------------------------

    def store_excel(
        self,
        document_key: str,
        document_name: str,
        dataframes: dict[str, pd.DataFrame],
        semantic_rows: list[dict[str, Any]] | None = None,
    ) -> str:
        """Create a SQLite database for an Excel file and register it.

        Args:
            document_key: Unique key for the document (e.g. "mayan:42").
            document_name: Human-readable filename (e.g. "budget.xlsx").
            dataframes: Mapping of sheet/table name to DataFrame.

        Returns:
            Path to the created SQLite database.
        """
        if not dataframes:
            logger.warning("No dataframes provided for %s — skipping SQLite creation", document_key)
            return ""

        # Sanitize document_key for use as a filename
        safe_key = re.sub(r"[^a-zA-Z0-9_-]", "_", document_key)
        db_path = self.db_dir / f"{safe_key}.db"

        # Remove existing DB to ensure a clean replacement
        if db_path.exists():
            db_path.unlink()

        total_rows = 0
        schema_info: dict[str, Any] = {}
        sheet_names: list[str] = []

        conn = sqlite3.connect(str(db_path))
        try:
            for table_name, df in dataframes.items():
                sql_table = _sanitize_table_name(table_name)
                sheet_names.append(sql_table)

                # Clean column names for SQL compatibility
                clean_df = df.copy()
                clean_df.columns = [_sanitize_column_name(c) for c in clean_df.columns]

                # Drop internal metadata columns
                if "__section__" in clean_df.columns:
                    clean_df = clean_df.drop(columns=["__section__"])

                clean_df.to_sql(sql_table, conn, if_exists="replace", index=False)
                total_rows += len(clean_df)

                # Build schema metadata for this table
                schema_info[sql_table] = _build_table_schema(clean_df, sql_table)

            if semantic_rows:
                facts_df = _build_facts_dataframe(semantic_rows)
                if not facts_df.empty:
                    facts_df.to_sql("excel_facts", conn, if_exists="replace", index=False)
                    schema_info["excel_facts"] = _build_table_schema(facts_df, "excel_facts")

            conn.commit()
        finally:
            conn.close()

        # Register in PostgreSQL or local fallback
        reg_entry = {
            "document_key": document_key,
            "document_name": document_name,
            "db_path": str(db_path),
            "schema_json": schema_info,
            "sheet_names": sheet_names,
            "total_rows": total_rows,
        }
        if self._pg_available:
            self._register_database(**reg_entry)
        self._local_registry[document_key] = reg_entry

        logger.info(
            "Created SQLite DB for %s at %s (%d tables, %d rows)",
            document_key, db_path, len(sheet_names), total_rows,
        )
        return str(db_path)

    def _register_database(
        self,
        document_key: str,
        document_name: str,
        db_path: str,
        schema_json: dict,
        sheet_names: list[str],
        total_rows: int,
    ):
        """Insert or update the registry entry in PostgreSQL."""
        with self._pg_connect() as conn:
            conn.execute(
                f"""
                INSERT INTO {self._pg_schema}.excel_databases (
                    document_key, document_name, db_path, schema_json,
                    sheet_names, total_rows, created_at
                )
                VALUES (%s, %s, %s, %s::jsonb, %s::jsonb, %s, now())
                ON CONFLICT (document_key) DO UPDATE SET
                    document_name = EXCLUDED.document_name,
                    db_path = EXCLUDED.db_path,
                    schema_json = EXCLUDED.schema_json,
                    sheet_names = EXCLUDED.sheet_names,
                    total_rows = EXCLUDED.total_rows,
                    created_at = now()
                """,
                (
                    document_key,
                    document_name,
                    db_path,
                    json.dumps(schema_json),
                    json.dumps(sheet_names),
                    total_rows,
                ),
            )

    # ------------------------------------------------------------------
    # Query classification
    # ------------------------------------------------------------------

    def get_registry_entry(self, document_key: str) -> dict[str, Any] | None:
        """Fetch the registry row for a document, or None if not found."""
        # Check local registry first
        if document_key in self._local_registry:
            return self._local_registry[document_key]

        if not self._pg_available:
            return None

        with self._pg_connect() as conn:
            return conn.execute(
                f"""
                SELECT document_key, document_name, db_path, schema_json,
                       sheet_names, total_rows, created_at
                FROM {self._pg_schema}.excel_databases
                WHERE document_key = %s
                """,
                (document_key,),
            ).fetchone()

    def has_sql_database(self, document_key: str) -> bool:
        """Check whether a SQLite database exists for the given document."""
        entry = self.get_registry_entry(document_key)
        if entry is None:
            return False
        return Path(entry["db_path"]).exists()

    def get_schema_description(self, document_key: str) -> str:
        """Return a human-readable schema description for the LLM prompt."""
        entry = self.get_registry_entry(document_key)
        if entry is None:
            return ""

        schema = entry["schema_json"]
        if isinstance(schema, str):
            schema = json.loads(schema)

        lines: list[str] = []
        for table_name, table_info in schema.items():
            lines.append(f"Table: {table_name}")
            lines.append(f"  Rows: {table_info.get('row_count', '?')}")
            for col in table_info.get("columns", []):
                col_line = f"  - {col['name']} ({col['dtype']})"
                # Show ALL values for text columns with low cardinality
                # so the LLM can write exact WHERE clauses
                unique_count = col.get("unique_count", 0)
                is_text = col["dtype"] in ("object", "str")
                if is_text and unique_count <= 30 and col.get("sample_values"):
                    col_line += f" -- ALL values: {col['sample_values']}"
                elif col.get("sample_values"):
                    col_line += f" -- e.g.: {col['sample_values']}"
                if col.get("min") is not None:
                    col_line += f" -- range: [{col['min']}, {col['max']}]"
                lines.append(col_line)
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # SQL execution (SELECT only)
    # ------------------------------------------------------------------

    def execute_sql(self, document_key: str, sql: str) -> list[dict[str, Any]]:
        """Execute a read-only SQL query against the document's SQLite DB.

        Args:
            document_key: The document to query.
            sql: A SELECT statement.

        Returns:
            List of result rows as dicts.

        Raises:
            ValueError: If the SQL is not a SELECT or the document has no DB.
            FileNotFoundError: If the SQLite file is missing on disk.
        """
        # Validate: only SELECT allowed
        stripped = sql.strip()
        if _MUTATION_PATTERN.match(stripped):
            raise ValueError(
                f"Only SELECT queries are allowed. Received: {stripped[:80]}..."
            )
        if not stripped.upper().startswith("SELECT"):
            raise ValueError(
                f"Query must start with SELECT. Received: {stripped[:80]}..."
            )

        entry = self.get_registry_entry(document_key)
        if entry is None:
            raise ValueError(f"No SQLite database registered for document_key={document_key!r}")

        db_path = Path(entry["db_path"])
        if not db_path.exists():
            raise FileNotFoundError(f"SQLite database file not found: {db_path}")

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(stripped)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def search_facts(
        self,
        document_key: str,
        query: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Search provenance-aware semantic facts stored for a document."""
        entry = self.get_registry_entry(document_key)
        if entry is None:
            return []

        db_path = Path(entry["db_path"])
        if not db_path.exists():
            return []

        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        try:
            exists = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='excel_facts'"
            ).fetchone()
            if not exists:
                return []

            rows = [dict(r) for r in conn.execute("SELECT * FROM excel_facts").fetchall()]
        finally:
            conn.close()

        scored = []
        for row in rows:
            score = _score_fact_row(query, row)
            if score > 0:
                row["_score"] = score
                scored.append(row)

        scored.sort(key=lambda item: (item["_score"], len(item.get("fact_text", ""))), reverse=True)
        return scored[:limit]

    def answer_provenance(
        self,
        document_key: str,
        query: str,
        limit: int = 3,
    ) -> dict[str, Any]:
        """Return a grounded answer for provenance-style questions."""
        matches = self.search_facts(document_key, query, limit=limit)
        if not matches:
            return {
                "question": query,
                "answer": "I couldn't find a grounded provenance match for that value in this workbook.",
                "results": [],
            }

        best = matches[0]
        source_cells = _load_json_field(best.get("source_cells_json"))
        source_labels = _load_json_field(best.get("source_labels_json"))
        column_paths = _load_json_field(best.get("column_paths_json"))
        section_path = _load_json_field(best.get("section_path_json"))
        row_semantic_texts = _load_json_field(best.get("row_semantic_texts_json"))
        display_sheet = (
            best.get("original_sheet_name")
            or _extract_original_sheet_name(row_semantic_texts)
            or best.get("sheet_name")
        )
        unique_labels = []
        if isinstance(source_labels, dict):
            unique_labels = list(dict.fromkeys(v for v in source_labels.values() if v))
        referenced_value = None
        value_match = re.search(r"\b\d+(?:\.\d+)?\b", query)
        if value_match:
            referenced_value = value_match.group(0)

        lead = "This value"
        if referenced_value:
            lead = f"The value {referenced_value}"

        sentence = [lead]
        if display_sheet:
            sentence.append(f"comes from the '{display_sheet}' sheet")
        if section_path:
            sentence.append(f"in the {' > '.join(section_path)} section")
        if unique_labels:
            sentence.append(f"for the line item '{unique_labels[0]}'")
        if column_paths:
            sentence.append(f"under {' / '.join(column_paths)}")
        sentence_text = " ".join(sentence) + "."

        details = []
        if best.get("units"):
            details.append(f"Units: {best['units']}.")
        formatted_cells = _format_source_cells(source_cells, source_labels)
        if formatted_cells:
            details.append(f"Source cells: {formatted_cells}.")
        if best.get("fact_text"):
            details.append(f"Grounded fact: {best['fact_text']}")

        return {
            "question": query,
            "answer": " ".join([sentence_text] + details),
            "results": matches,
            "semantic_rows": row_semantic_texts,
        }

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def remove_excel(self, document_key: str):
        """Remove a document's SQLite DB and its registry entry."""
        entry = self.get_registry_entry(document_key)
        if entry:
            db_path = Path(entry["db_path"])
            if db_path.exists():
                db_path.unlink()
                logger.info("Deleted SQLite DB at %s", db_path)

        with self._pg_connect() as conn:
            conn.execute(
                f"DELETE FROM {self._pg_schema}.excel_databases WHERE document_key = %s",
                (document_key,),
            )

    def list_databases(self) -> list[dict[str, Any]]:
        """List all registered Excel databases."""
        if self._pg_available:
            with self._pg_connect() as conn:
                return conn.execute(
                    f"""
                    SELECT document_key, document_name, db_path, sheet_names,
                           total_rows, created_at
                    FROM {self._pg_schema}.excel_databases
                    ORDER BY created_at DESC
                    """
                ).fetchall()
        # Fallback to local registry
        return list(self._local_registry.values())

    def build_catalog(self) -> str:
        """Build a compact catalog of ALL SQL databases for LLM routing.

        Returns a text summary like:
            DB 1: "Financial Statements.xlsx" (key=mayan:10)
              Tables: income_statement (13 rows: Revenue, Cost of revenue, R&D, ...)
                      balance_sheet (16 rows: Cash, Investments, Assets, ...)
            DB 2: "Apple.xlsx" (key=mayan:11)
              Tables: income_statement (37 rows: Products, Services, Total net sales, ...)
        """
        all_keys = []
        if self._pg_available:
            try:
                dbs = self.list_databases()
                all_keys = [(d["document_key"], d["document_name"]) for d in dbs]
            except Exception:
                pass
        for key, entry in self._local_registry.items():
            if key not in [k for k, _ in all_keys]:
                all_keys.append((key, entry.get("document_name", key)))

        if not all_keys:
            return ""

        lines = []
        for i, (doc_key, doc_name) in enumerate(all_keys, 1):
            entry = self.get_registry_entry(doc_key)
            if not entry:
                continue

            schema = entry["schema_json"]
            if isinstance(schema, str):
                schema = json.loads(schema)

            lines.append(f'DB {i}: "{doc_name}" (key={doc_key})')
            for table_name, table_info in schema.items():
                row_count = table_info.get("row_count", "?")
                cols = table_info.get("columns", [])
                col_names = [c["name"] for c in cols]
                # Show sample values from text columns for context
                sample_vals = []
                for c in cols:
                    if c["dtype"] in ("object", "str") and c.get("sample_values"):
                        sample_vals.extend(str(v) for v in c["sample_values"][:4])
                sample_str = ", ".join(sample_vals[:6]) if sample_vals else ""
                lines.append(f"  - {table_name} ({row_count} rows, cols: {col_names})")
                if sample_str:
                    lines.append(f"    Sample values: {sample_str}")

        return "\n".join(lines)


# ======================================================================
# Query classifier (module-level function)
# ======================================================================

def classify_query(query: str) -> str:
    """Classify whether a query needs SQL computation or semantic retrieval.

    Returns:
        "sql" if the question involves computation, aggregation, filtering,
        or comparison that is best answered by running a SQL query.
        "semantic" if the question is looking for textual information,
        explanations, or descriptions.
    """
    if not query or not query.strip():
        return "semantic"

    text = query.strip()

    if _PROVENANCE_SIGNAL_WORDS.search(text):
        return "provenance"

    # Count signal matches
    sql_matches = len(_SQL_SIGNAL_WORDS.findall(text))
    semantic_matches = len(_SEMANTIC_SIGNAL_WORDS.findall(text))

    # Strong SQL signals: numbers referenced or comparison operators present
    has_numbers = bool(re.search(r"\b\d+(?:\.\d+)?\b", text))
    has_comparison_ops = bool(re.search(r"[<>=!]{1,2}\s*\d", text))

    if has_comparison_ops:
        sql_matches += 2

    if has_numbers and sql_matches > 0:
        sql_matches += 1

    # "What is the total/sum/average/max/min ..." — the question frame
    # ("what is") is just phrasing; the core intent is the aggregation word.
    # Discount the semantic signal from question frames when a strong SQL
    # aggregation keyword is present.
    _question_frame = re.compile(
        r"^(what is|what are|how much|how many)\b",
        re.IGNORECASE,
    )
    _strong_sql = re.compile(
        r"\b(sum|total|average|avg|mean|count|max|maximum|min|minimum|"
        r"highest|lowest|how many|how much|top|bottom|rank|filter|group)\b",
        re.IGNORECASE,
    )
    if _question_frame.search(text) and _strong_sql.search(text):
        sql_matches += 1  # boost SQL to break the tie

    # If only SQL signals and no semantic signals, it is SQL
    if sql_matches > 0 and semantic_matches == 0:
        return "sql"

    # If SQL signals dominate
    if sql_matches > semantic_matches:
        return "sql"

    # If semantic signals dominate or tie, prefer semantic
    return "semantic"


# ======================================================================
# Helpers
# ======================================================================

def _sanitize_table_name(name: str) -> str:
    """Convert a sheet/table name to a valid SQL identifier."""
    clean = re.sub(r"[^a-zA-Z0-9_]", "_", name.strip())
    clean = re.sub(r"_+", "_", clean).strip("_").lower()
    if not clean or clean[0].isdigit():
        clean = "t_" + clean
    # Avoid SQLite reserved words
    if clean.upper() in {"TABLE", "INDEX", "SELECT", "ORDER", "GROUP", "WHERE", "FROM"}:
        clean = "t_" + clean
    return clean


def _build_facts_dataframe(semantic_rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Normalize Stage 2 semantic rows into a facts table."""
    records: list[dict[str, Any]] = []
    for row in semantic_rows:
        metadata = row.get("metadata", {}) or {}
        records.append(
            {
                "sheet_name": row.get("sheet_name"),
                "original_sheet_name": metadata.get("original_sheet_name"),
                "row_index": row.get("row_index"),
                "section": metadata.get("section"),
                "section_path_json": json.dumps(metadata.get("section_path", [])),
                "column_paths_json": json.dumps(metadata.get("column_paths", [])),
                "units": metadata.get("units"),
                "source_cells_json": json.dumps(metadata.get("source_cells", {})),
                "source_labels_json": json.dumps(metadata.get("source_labels", {})),
                "row_semantic_texts_json": json.dumps(metadata.get("row_semantic_texts", [])),
                "fact_text": row.get("text", ""),
            }
        )
    return pd.DataFrame(records)


def _load_json_field(value: Any) -> Any:
    """Load a JSON-encoded sqlite field when possible."""
    if value in (None, ""):
        return [] if value == "[]" else {}
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except Exception:
        return value


def _extract_original_sheet_name(row_semantic_texts: Any) -> str | None:
    """Pull the original sheet name out of Stage 1 semantic text if available."""
    if not isinstance(row_semantic_texts, list) or not row_semantic_texts:
        return None
    first = str(row_semantic_texts[0])
    match = re.search(r"Sheet:\s*([^|]+)", first)
    if not match:
        return None
    return match.group(1).strip()


def _format_source_cells(source_cells: Any, source_labels: Any) -> str:
    """Render source cells using human labels when available."""
    if not isinstance(source_cells, dict) or not source_cells:
        return ""

    parts = []
    for raw_label, address in source_cells.items():
        label = raw_label
        if isinstance(source_labels, dict):
            label = source_labels.get(raw_label) or raw_label
        parts.append(f"{label} ({address})")
    return ", ".join(parts)


def _score_fact_row(query: str, row: dict[str, Any]) -> int:
    """Simple lexical scorer for provenance facts."""
    query_lower = query.lower()
    terms = [term for term in re.findall(r"[a-z0-9]+", query_lower) if len(term) > 1]
    score = 0

    fact_text = str(row.get("fact_text", "")).lower()
    if not fact_text:
        return 0

    for term in terms:
        if term in fact_text:
            score += 3

    for number in re.findall(r"\b\d+(?:\.\d+)?\b", query_lower):
        if number in fact_text:
            score += 8

    if _PROVENANCE_SIGNAL_WORDS.search(query_lower):
        score += 2

    labels = _load_json_field(row.get("source_labels_json"))
    if isinstance(labels, dict):
        for label in labels.values():
            label_text = str(label).lower()
            if label_text and label_text in query_lower:
                score += 10

    section = str(row.get("section", "")).lower()
    if section and section in query_lower:
        score += 5

    return score


def _sanitize_column_name(name: str) -> str:
    """Convert a column name to a valid SQL identifier."""
    if not isinstance(name, str):
        name = str(name)
    clean = re.sub(r"[^a-zA-Z0-9_]", "_", name.strip())
    clean = re.sub(r"_+", "_", clean).strip("_").lower()
    if not clean:
        clean = "col"
    if clean[0].isdigit():
        clean = "c_" + clean
    return clean


def _build_table_schema(df: pd.DataFrame, table_name: str) -> dict[str, Any]:
    """Build a JSON-serializable schema description for a DataFrame/table."""
    columns: list[dict[str, Any]] = []
    for col in df.columns:
        dtype = str(df[col].dtype)
        non_null = int(df[col].notna().sum())
        unique_vals = df[col].dropna().unique()

        # For text columns with low cardinality, store ALL values
        # so the LLM can write exact WHERE clauses
        is_text = dtype in ("object", "str")
        if is_text and len(unique_vals) <= 30:
            sample = [_to_native(v) for v in unique_vals.tolist()]
        else:
            sample = [_to_native(v) for v in unique_vals[:5].tolist()]

        col_info: dict[str, Any] = {
            "name": col,
            "dtype": dtype,
            "non_null_count": non_null,
            "total_count": len(df),
            "unique_count": len(unique_vals),
            "sample_values": sample,
        }

        # Add range for numeric columns
        if df[col].dtype.kind in ("i", "f"):
            col_info["min"] = _to_native(df[col].min())
            col_info["max"] = _to_native(df[col].max())

        columns.append(col_info)

    return {
        "table_name": table_name,
        "row_count": len(df),
        "column_count": len(df.columns),
        "columns": columns,
    }


def _to_native(value: Any) -> Any:
    """Convert numpy/pandas scalar to a native Python type for JSON."""
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, float) and (value != value):  # NaN check
        return None
    return value
