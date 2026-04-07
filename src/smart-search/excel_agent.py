"""LLM-powered SQL agent for natural language queries over Excel data.

Takes a natural language question, generates SQLite-compatible SQL using the
auto-generated schema description, executes it, and synthesises a readable answer.

Provider fallback chain: Groq -> Ollama -> Claude -> OpenAI  (mirrors app.py)
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Callable

import httpx

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dangerous SQL keywords (anything that mutates state)
# ---------------------------------------------------------------------------
# DML/DDL statements that must be blocked (mutate data or schema)
_UNSAFE_STATEMENTS: set[str] = {
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "ALTER",
    "CREATE",
    "TRUNCATE",
    "MERGE",
    "GRANT",
    "REVOKE",
    "ATTACH",
    "DETACH",
}

# Keywords that are unsafe as STATEMENTS but safe as FUNCTIONS:
#   REPLACE(x, y, z) = safe string function
#   REPLACE INTO ... = unsafe DML
#   EXEC/EXECUTE = unsafe as standalone, but okay inside function names
# We match these only when NOT followed by '(' (i.e., not a function call)
_UNSAFE_WHEN_NOT_FUNCTION: set[str] = {
    "REPLACE",
    "EXEC",
    "EXECUTE",
}

# Pattern for always-unsafe keywords (word boundary match)
_UNSAFE_PATTERN = re.compile(
    r"\b(" + "|".join(_UNSAFE_STATEMENTS) + r")\b",
    re.IGNORECASE,
)

# Pattern for keywords that are only unsafe when NOT used as functions
_UNSAFE_NON_FUNC_PATTERN = re.compile(
    r"\b(" + "|".join(_UNSAFE_WHEN_NOT_FUNCTION) + r")\b(?!\s*\()",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------
_SQL_SYSTEM_PROMPT_TEMPLATE = """\
You are a data analyst. Answer questions by writing SQLite-compatible SELECT queries.

DATABASE SCHEMA:
{schema_description}

Column role annotations help you write better queries:
- [role: metric] columns are numeric measures — use SUM, AVG, MIN, MAX on these
- [role: categorical] columns are dimensions — use these in GROUP BY and WHERE filters
- [role: temporal] columns represent time — use these for date filtering and ordering
- [role: identifier] columns uniquely identify rows — use these in WHERE for lookups
- [role: computed] columns are derived values — avoid aggregating these unless asked

Rules:
- Write SELECT queries only
- Use exact column names from the schema (wrap column names containing spaces in double quotes)
- For date filtering, use formats shown in the schema
- Prefer GROUP BY on dimension columns when aggregating metrics
- Return ONLY the SQL query, no explanation
- Do NOT wrap the SQL in markdown code fences
"""

_FIX_SYSTEM_PROMPT = """\
You are a SQL debugging expert. Fix the following SQLite query that produced an error.
Return ONLY the corrected SQL query, no explanation.
Do NOT wrap the SQL in markdown code fences.
"""

_ANSWER_SYSTEM_PROMPT = """\
You are a data analyst answering a user's question about spreadsheet data.
Use the query results provided to give a clear, concise answer.
Cite specific numbers and values from the data.
If the results are empty, say the data does not contain matching records.
"""

_ROUTE_SYSTEM_PROMPT = """\
You are a database router. Given a user question and a catalog of available databases, \
pick the ONE database most likely to contain the answer.

Return ONLY the database key (e.g., "mayan:10"). No explanation."""


class ExcelSQLAgent:
    """Generate SQL from natural language, execute it, and synthesise answers.

    Parameters
    ----------
    max_retries : int
        Maximum attempts to fix a failing SQL query (default 3).
    timeout : int
        HTTP timeout in seconds for LLM API calls (default 30).
    """

    def __init__(self, max_retries: int = 3, timeout: int = 30) -> None:
        self.max_retries = max_retries
        self.timeout = timeout

    # ------------------------------------------------------------------
    # Database routing (pick the right DB from multiple)
    # ------------------------------------------------------------------

    def route_to_database(self, question: str, catalog: str, doc_keys: list[str]) -> str | None:
        """Given a question and a catalog of databases, pick the best one.

        Uses a single lightweight LLM call to select from available databases.
        Falls back to the first key if LLM is unavailable.

        Returns the document_key of the best-matching database.
        """
        if not doc_keys:
            return None
        if len(doc_keys) == 1:
            return doc_keys[0]

        user_prompt = (
            f"Question: {question}\n\n"
            f"Available databases:\n{catalog}\n\n"
            f"Which database key should I query? Return ONLY the key."
        )

        raw = self._call_llm(_ROUTE_SYSTEM_PROMPT, user_prompt)
        if raw:
            picked = raw.strip().strip('"').strip("'")
            # Find the closest match from available keys
            for key in doc_keys:
                if key in picked:
                    logger.info("Router picked %s for question: %s", key, question[:80])
                    return key

        # Fallback: first key
        logger.info("Router fallback to %s", doc_keys[0])
        return doc_keys[0]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_sql(self, question: str, schema: str) -> str:
        """Send the question + schema to the LLM and return a SQL string."""
        system_prompt = _SQL_SYSTEM_PROMPT_TEMPLATE.format(
            schema_description=schema,
        )
        user_prompt = question

        sql = self._call_llm(system_prompt, user_prompt)
        if sql is None:
            raise RuntimeError(
                "All LLM providers failed. Check API keys and connectivity."
            )
        return self._clean_sql(sql)

    def validate_sql(self, sql: str) -> bool:
        """Return True only if *sql* is a safe read-only SELECT statement."""
        stripped = sql.strip().rstrip(";").strip()

        # Must start with SELECT (or WITH for CTEs)
        if not re.match(r"^\s*(SELECT|WITH)\b", stripped, re.IGNORECASE):
            return False

        # Reject unsafe keywords (check outside string literals)
        sql_without_strings = re.sub(r"'[^']*'", "''", stripped)
        if _UNSAFE_PATTERN.search(sql_without_strings):
            return False
        # Reject REPLACE/EXEC only when used as statements, not functions
        if _UNSAFE_NON_FUNC_PATTERN.search(sql_without_strings):
            return False

        # Block multiple statements (semicolons outside string literals)
        if ";" in sql_without_strings:
            return False

        return True

    def ask(
        self,
        question: str,
        schema: str,
        execute_fn: Callable[[str], list[dict[str, Any]]],
    ) -> dict[str, Any]:
        """End-to-end: question in, answer dict out.

        Parameters
        ----------
        question : str
            Natural language question about the data.
        schema : str
            Auto-generated schema description (from ExcelEnricher).
        execute_fn : callable
            ``execute_fn(sql) -> list[dict]``  — executes SQL against the
            SQLite database and returns rows as dicts.

        Returns
        -------
        dict with keys:
            question, sql, results, answer, attempts, model
        """
        last_error: str | None = None
        sql: str | None = None

        for attempt in range(1, self.max_retries + 1):
            # --- Generate or fix SQL ---
            if last_error is None:
                sql = self.generate_sql(question, schema)
            else:
                sql = self._fix_sql(sql, last_error, schema)

            # --- Safety gate ---
            if not self.validate_sql(sql):
                logger.warning("Rejected unsafe SQL: %s", sql)
                return {
                    "question": question,
                    "sql": sql,
                    "results": [],
                    "answer": "The generated query was rejected for safety reasons. Only SELECT queries are allowed.",
                    "attempts": attempt,
                    "error": "unsafe_sql",
                }

            # --- Execute ---
            try:
                results = execute_fn(sql)
                answer, model = self._synthesise_answer(question, sql, results)
                return {
                    "question": question,
                    "sql": sql,
                    "results": results,
                    "answer": answer,
                    "attempts": attempt,
                    "model": model,
                }
            except Exception as exc:
                last_error = str(exc)
                logger.info(
                    "SQL attempt %d/%d failed: %s — SQL: %s",
                    attempt,
                    self.max_retries,
                    last_error,
                    sql,
                )

        # All retries exhausted
        return {
            "question": question,
            "sql": sql,
            "results": [],
            "answer": f"Unable to answer after {self.max_retries} attempts. Last error: {last_error}",
            "attempts": self.max_retries,
            "error": last_error,
        }

    # ------------------------------------------------------------------
    # Internal — LLM provider chain (sync httpx, mirrors app.py pattern)
    # ------------------------------------------------------------------

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str | None:
        """Try each provider in order; return first successful response."""
        providers: list[tuple[str, Callable]] = [
            ("Groq", self._call_groq),
            ("Ollama", self._call_ollama),
            ("Anthropic", self._call_anthropic),
            ("OpenAI", self._call_openai),
        ]
        for name, call_fn in providers:
            try:
                text = call_fn(system_prompt, user_prompt)
                if text:
                    logger.debug("LLM response from %s: %s", name, text[:200])
                    return text
            except Exception:
                logger.debug("Provider %s failed, trying next", name, exc_info=True)
                continue
        return None

    def _call_llm_with_model(
        self, system_prompt: str, user_prompt: str
    ) -> tuple[str | None, str | None]:
        """Like _call_llm but also returns the model name used."""
        providers: list[tuple[str, Callable]] = [
            ("Groq (llama-3.3-70b)", self._call_groq),
            ("Ollama (llama3.2)", self._call_ollama),
            ("Claude Haiku 4.5", self._call_anthropic),
            ("GPT-4o Mini", self._call_openai),
        ]
        for model_name, call_fn in providers:
            try:
                text = call_fn(system_prompt, user_prompt)
                if text:
                    return text, model_name
            except Exception:
                continue
        return None, None

    # --- Groq ----------------------------------------------------------

    def _call_groq(self, system_prompt: str, user_prompt: str) -> str | None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            return None
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "max_tokens": 500,
                    "temperature": 0,
                },
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    # --- Ollama --------------------------------------------------------

    def _call_ollama(self, system_prompt: str, user_prompt: str) -> str | None:
        with httpx.Client(timeout=60) as client:
            resp = client.post(
                "http://localhost:11434/api/chat",
                json={
                    "model": "llama3.2",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "stream": False,
                },
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            return data.get("message", {}).get("content")

    # --- Anthropic (Claude) -------------------------------------------

    def _call_anthropic(self, system_prompt: str, user_prompt: str) -> str | None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            return None
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": "claude-haiku-4-5-20251001",
                    "max_tokens": 500,
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": user_prompt}],
                },
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            return data.get("content", [{}])[0].get("text")

    # --- OpenAI -------------------------------------------------------

    def _call_openai(self, system_prompt: str, user_prompt: str) -> str | None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return None
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "max_tokens": 500,
                    "temperature": 0,
                },
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            return data["choices"][0]["message"]["content"]

    # ------------------------------------------------------------------
    # Internal — SQL cleanup / retry / synthesis
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_sql(raw: str) -> str:
        """Strip markdown fences and leading/trailing whitespace from LLM output."""
        cleaned = raw.strip()

        # Remove ```sql ... ``` wrappers
        if cleaned.startswith("```"):
            # Drop first line (```sql or ```)
            lines = cleaned.split("\n")
            lines = lines[1:]
            # Drop trailing ```
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            cleaned = "\n".join(lines).strip()

        return cleaned

    def _fix_sql(self, broken_sql: str, error: str, schema: str) -> str:
        """Ask the LLM to fix a SQL query that produced an error."""
        user_prompt = (
            f"Original query:\n{broken_sql}\n\n"
            f"Error:\n{error}\n\n"
            f"Schema:\n{schema}\n\n"
            "Fix this SQL error and return ONLY the corrected query."
        )
        fixed = self._call_llm(_FIX_SYSTEM_PROMPT, user_prompt)
        if fixed is None:
            # If LLM is completely unavailable, return original (will fail again)
            return broken_sql
        return self._clean_sql(fixed)

    def _synthesise_answer(
        self,
        question: str,
        sql: str,
        results: list[dict[str, Any]],
    ) -> tuple[str, str | None]:
        """Call LLM to turn raw query results into a natural language answer.

        Returns (answer_text, model_name).
        """
        if not results:
            return "The query returned no results.", None

        # Truncate large result sets so we don't blow up the context window
        max_rows = 50
        truncated = len(results) > max_rows
        display_results = results[:max_rows]

        results_text = _format_results(display_results)
        if truncated:
            results_text += f"\n... ({len(results) - max_rows} more rows omitted)"

        user_prompt = (
            f"Question: {question}\n\n"
            f"SQL query used: {sql}\n\n"
            f"Query results:\n{results_text}\n\n"
            "Provide a clear, concise answer citing the data."
        )

        answer, model = self._call_llm_with_model(
            _ANSWER_SYSTEM_PROMPT, user_prompt
        )
        if answer is None:
            # Graceful degradation: return raw results as the answer
            return f"Query results:\n{results_text}", None
        return answer, model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_results(rows: list[dict[str, Any]]) -> str:
    """Format a list of row-dicts into a readable text table."""
    if not rows:
        return "(no rows)"

    columns = list(rows[0].keys())
    lines = [" | ".join(str(c) for c in columns)]
    lines.append("-" * len(lines[0]))

    for row in rows:
        lines.append(" | ".join(str(row.get(c, "")) for c in columns))

    return "\n".join(lines)
