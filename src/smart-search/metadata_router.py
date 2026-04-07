"""Metadata Router — two-stage document routing for RAG.

Stage 1: Find the right document(s) using metadata (keywords + vectors + RRF)
Stage 2: Search chunks only within the winning documents

Uses a dedicated `rag.document_routes` table with:
- keywords (tsvector for PostgreSQL full-text search)
- summary embedding (768d vector for semantic search)
- full hierarchy metadata (cabinet_ids, document_id, full_path)

Architecture:
    Query → [keyword search + vector search] → RRF → cross-encoder → top 5 docs
    → scoped chunk search within those 5 docs → answer
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import psycopg
from psycopg.rows import dict_row

logger = logging.getLogger(__name__)


class MetadataRouter:
    """Routes queries to the right documents before chunk search."""

    def __init__(self):
        self._pg_host = os.environ.get("RAG_PGHOST", os.environ.get("POSTGRES_HOST", "postgresql"))
        self._pg_port = int(os.environ.get("RAG_PGPORT", os.environ.get("POSTGRES_PORT", "5432")))
        self._pg_dbname = os.environ.get("RAG_PGDATABASE", os.environ.get("POSTGRES_DB", "mayan"))
        self._pg_user = os.environ.get("RAG_PGUSER", os.environ.get("POSTGRES_USER", "mayan"))
        self._pg_password = os.environ.get("RAG_PGPASSWORD", os.environ.get("POSTGRES_PASSWORD", "mayan"))
        self._pg_schema = os.environ.get("RAG_PG_SCHEMA", "rag")

        self._pg_available = False
        try:
            self._ensure_table()
            self._pg_available = True
        except Exception as e:
            logger.warning("MetadataRouter: PostgreSQL not available: %s", e)

    def _connect(self):
        return psycopg.connect(
            host=self._pg_host, port=self._pg_port,
            dbname=self._pg_dbname, user=self._pg_user,
            password=self._pg_password, row_factory=dict_row,
        )

    def _ensure_table(self):
        with self._connect() as conn:
            conn.execute(f"CREATE SCHEMA IF NOT EXISTS {self._pg_schema}")
            conn.execute(f"""
                CREATE TABLE IF NOT EXISTS {self._pg_schema}.document_routes (
                    document_key text PRIMARY KEY,
                    document_id integer,
                    document_name text NOT NULL,
                    company_name text,
                    company_id integer,
                    cabinet_id integer,
                    cabinet_path text,
                    cabinet_ids jsonb DEFAULT '[]'::jsonb,
                    full_path_ids jsonb DEFAULT '[]'::jsonb,
                    full_path text,
                    keywords text[] NOT NULL DEFAULT '{{}}',
                    keywords_text text NOT NULL DEFAULT '',
                    keywords_tsv tsvector GENERATED ALWAYS AS (
                        to_tsvector('english', keywords_text)
                    ) STORED,
                    summary_text text NOT NULL DEFAULT '',
                    summary_embedding vector(768),
                    chunk_count integer DEFAULT 0,
                    created_at timestamptz DEFAULT now(),
                    updated_at timestamptz DEFAULT now()
                )
            """)
            conn.execute(f"""
                CREATE INDEX IF NOT EXISTS doc_routes_keywords_gin
                ON {self._pg_schema}.document_routes USING GIN (keywords_tsv)
            """)
            conn.execute(f"""
                CREATE INDEX IF NOT EXISTS doc_routes_embedding_hnsw
                ON {self._pg_schema}.document_routes
                USING hnsw (summary_embedding vector_cosine_ops)
            """)
            conn.execute(f"""
                CREATE INDEX IF NOT EXISTS doc_routes_cabinet_ids_gin
                ON {self._pg_schema}.document_routes USING GIN (cabinet_ids)
            """)

    # ------------------------------------------------------------------
    # Index: store a document's route metadata
    # ------------------------------------------------------------------

    def store_route(
        self,
        document_key: str,
        document_id: int | None,
        document_name: str,
        keywords: list[str],
        summary_text: str,
        summary_embedding: list[float],
        cabinet_landmark: dict | None = None,
        chunk_count: int = 0,
    ) -> None:
        """Store or update the route metadata for a document."""
        if not self._pg_available:
            return

        landmark = cabinet_landmark or {}
        keywords_text = " ".join(keywords)

        with self._connect() as conn:
            conn.execute(f"""
                INSERT INTO {self._pg_schema}.document_routes (
                    document_key, document_id, document_name,
                    company_name, company_id, cabinet_id, cabinet_path,
                    cabinet_ids, full_path_ids, full_path,
                    keywords, keywords_text, summary_text, summary_embedding,
                    chunk_count, updated_at
                ) VALUES (
                    %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s,
                    %s, %s, %s, %s::vector, %s, now()
                )
                ON CONFLICT (document_key) DO UPDATE SET
                    document_name = EXCLUDED.document_name,
                    company_name = EXCLUDED.company_name,
                    company_id = EXCLUDED.company_id,
                    cabinet_id = EXCLUDED.cabinet_id,
                    cabinet_path = EXCLUDED.cabinet_path,
                    cabinet_ids = EXCLUDED.cabinet_ids,
                    full_path_ids = EXCLUDED.full_path_ids,
                    full_path = EXCLUDED.full_path,
                    keywords = EXCLUDED.keywords,
                    keywords_text = EXCLUDED.keywords_text,
                    summary_text = EXCLUDED.summary_text,
                    summary_embedding = EXCLUDED.summary_embedding,
                    chunk_count = EXCLUDED.chunk_count,
                    updated_at = now()
            """, (
                document_key, document_id, document_name,
                landmark.get("company_name"), landmark.get("company_id"),
                landmark.get("cabinet_id"), landmark.get("cabinet_path"),
                json.dumps(landmark.get("cabinet_ids", [])),
                json.dumps(landmark.get("full_path_ids", [])),
                landmark.get("full_path"),
                keywords, keywords_text, summary_text, summary_embedding,
                chunk_count,
            ))

        logger.info(
            "Stored route for %s: %d keywords, %d chars summary",
            document_key, len(keywords), len(summary_text),
        )

    # ------------------------------------------------------------------
    # Route: find the best documents for a query
    # ------------------------------------------------------------------

    def route(
        self,
        query: str,
        query_embedding: list[float],
        top_k: int = 5,
        cabinet_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """Find the top-K most relevant documents for a query.

        Uses hybrid search: keyword (tsvector) + vector (embedding) + RRF fusion.

        Returns list of dicts with document_key, score, and full metadata.
        """
        if not self._pg_available:
            return []

        schema = self._pg_schema

        # Cabinet scope filter
        cab_filter = ""
        params_keyword: list[Any] = [query]
        params_vector: list[Any] = [query_embedding]

        if cabinet_id:
            cab_filter = f"AND cabinet_ids @> %s::jsonb"
            params_keyword.append(json.dumps([cabinet_id]))
            params_vector.append(json.dumps([cabinet_id]))

        with self._connect() as conn:
            # Keyword search (tsvector) — use OR between terms so partial matches work
            # Split query into words, join with ' | ' for OR semantics
            import re as _re
            query_words = [w for w in _re.findall(r'\w+', query.lower()) if len(w) > 2]
            ts_query_str = " | ".join(query_words) if query_words else query

            keyword_results = conn.execute(f"""
                SELECT document_key,
                       ts_rank_cd(keywords_tsv, to_tsquery('english', %s)) as score
                FROM {schema}.document_routes
                WHERE keywords_tsv @@ to_tsquery('english', %s)
                {cab_filter}
                ORDER BY score DESC
                LIMIT 20
            """, [ts_query_str, ts_query_str] + (params_keyword[1:] if cabinet_id else [])).fetchall()

            # Vector search (embedding)
            vector_params = [query_embedding]
            if cabinet_id:
                vector_params.append(json.dumps([cabinet_id]))
            vector_results = conn.execute(f"""
                SELECT document_key,
                       1 - (summary_embedding <=> %s::vector) as score
                FROM {schema}.document_routes
                WHERE summary_embedding IS NOT NULL
                {cab_filter}
                ORDER BY summary_embedding <=> %s::vector
                LIMIT 20
            """, vector_params + [query_embedding] + ([json.dumps([cabinet_id])] if cabinet_id else [])).fetchall()

            # RRF Fusion
            k = 60  # RRF constant
            doc_scores: dict[str, float] = {}

            for rank, row in enumerate(keyword_results):
                doc_scores[row["document_key"]] = doc_scores.get(row["document_key"], 0) + 1.0 / (k + rank + 1)

            for rank, row in enumerate(vector_results):
                doc_scores[row["document_key"]] = doc_scores.get(row["document_key"], 0) + 1.0 / (k + rank + 1)

            # Sort by fused score
            ranked = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

            if not ranked:
                return []

            # Fetch full metadata for winners
            winner_keys = [r[0] for r in ranked]
            placeholders = ", ".join(["%s"] * len(winner_keys))
            routes = conn.execute(f"""
                SELECT * FROM {schema}.document_routes
                WHERE document_key IN ({placeholders})
            """, winner_keys).fetchall()

            # Build result with RRF scores
            route_map = {r["document_key"]: r for r in routes}
            results = []
            for doc_key, rrf_score in ranked:
                if doc_key in route_map:
                    route = dict(route_map[doc_key])
                    route["rrf_score"] = rrf_score
                    # Remove embedding from response (too large)
                    route.pop("summary_embedding", None)
                    results.append(route)

            return results

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def list_routes(self) -> list[dict]:
        """List all stored routes."""
        if not self._pg_available:
            return []
        with self._connect() as conn:
            rows = conn.execute(f"""
                SELECT document_key, document_name, company_name, cabinet_path,
                       full_path, array_length(keywords, 1) as keyword_count,
                       chunk_count, updated_at
                FROM {self._pg_schema}.document_routes
                ORDER BY document_name
            """).fetchall()
            return [dict(r) for r in rows]

    def get_route(self, document_key: str) -> dict | None:
        """Get a single route by document key."""
        if not self._pg_available:
            return None
        with self._connect() as conn:
            row = conn.execute(f"""
                SELECT * FROM {self._pg_schema}.document_routes
                WHERE document_key = %s
            """, (document_key,)).fetchone()
            if row:
                r = dict(row)
                r.pop("summary_embedding", None)
                return r
            return None

    def update_cabinet(self, document_key: str, cabinet_landmark: dict) -> None:
        """Update just the cabinet info on an existing route."""
        if not self._pg_available or not cabinet_landmark:
            return
        with self._connect() as conn:
            conn.execute(f"""
                UPDATE {self._pg_schema}.document_routes SET
                    company_name = %s, company_id = %s,
                    cabinet_id = %s, cabinet_path = %s,
                    cabinet_ids = %s::jsonb, full_path_ids = %s::jsonb,
                    full_path = %s, updated_at = now()
                WHERE document_key = %s
            """, (
                cabinet_landmark.get("company_name"),
                cabinet_landmark.get("company_id"),
                cabinet_landmark.get("cabinet_id"),
                cabinet_landmark.get("cabinet_path"),
                json.dumps(cabinet_landmark.get("cabinet_ids", [])),
                json.dumps(cabinet_landmark.get("full_path_ids", [])),
                cabinet_landmark.get("full_path"),
                document_key,
            ))
        logger.info("Updated cabinet info for %s", document_key)

    def delete_route(self, document_key: str) -> None:
        """Delete a route."""
        if not self._pg_available:
            return
        with self._connect() as conn:
            conn.execute(f"""
                DELETE FROM {self._pg_schema}.document_routes WHERE document_key = %s
            """, (document_key,))
