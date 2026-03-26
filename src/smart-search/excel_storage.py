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
                if col.get("sample_values"):
                    col_line += f" -- e.g.: {col['sample_values']}"
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
        with self._pg_connect() as conn:
            return conn.execute(
                f"""
                SELECT document_key, document_name, db_path, sheet_names,
                       total_rows, created_at
                FROM {self._pg_schema}.excel_databases
                ORDER BY created_at DESC
                """
            ).fetchall()


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

        # Sample values for LLM context (up to 5)
        sample = unique_vals[:5].tolist()
        # Convert numpy/pandas types to native Python
        sample = [_to_native(v) for v in sample]

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
