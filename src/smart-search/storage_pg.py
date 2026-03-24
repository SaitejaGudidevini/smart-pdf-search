"""PostgreSQL + pgvector storage backend for RAG chunks."""

from __future__ import annotations

import json
import os
import re
import uuid
from typing import Any

import psycopg
from psycopg.rows import dict_row


class PostgresVectorStore:
    """Stores RAG chunks in PostgreSQL using the pgvector extension."""

    def __init__(self):
        self.host = os.environ.get("RAG_PGHOST", os.environ.get("POSTGRES_HOST", "postgresql"))
        self.port = int(os.environ.get("RAG_PGPORT", os.environ.get("POSTGRES_PORT", "5432")))
        self.dbname = os.environ.get("RAG_PGDATABASE", os.environ.get("POSTGRES_DB", "mayan"))
        self.user = os.environ.get("RAG_PGUSER", os.environ.get("POSTGRES_USER", "mayan"))
        self.password = os.environ.get("RAG_PGPASSWORD", os.environ.get("POSTGRES_PASSWORD", "mayan"))
        self.schema = os.environ.get("RAG_PG_SCHEMA", "rag")
        self._ensure_schema()

    def _connect(self):
        return psycopg.connect(
            host=self.host,
            port=self.port,
            dbname=self.dbname,
            user=self.user,
            password=self.password,
            row_factory=dict_row,
        )

    def _ensure_schema(self):
        with self._connect() as conn:
            conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            conn.execute(f"CREATE SCHEMA IF NOT EXISTS {self.schema}")
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.schema}.chunks (
                    id uuid PRIMARY KEY,
                    document_key text NOT NULL,
                    mayan_doc_id integer,
                    document_name text NOT NULL,
                    page_number integer NOT NULL,
                    chunk_type text NOT NULL CHECK (chunk_type IN ('parent', 'child')),
                    parent_id text,
                    section text,
                    doc_type text,
                    start_line integer,
                    end_line integer,
                    content text NOT NULL,
                    enriched_content text,
                    lines jsonb NOT NULL DEFAULT '[]'::jsonb,
                    metadata jsonb NOT NULL DEFAULT '{{}}'::jsonb,
                    embedding vector(384),
                    content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
                    created_at timestamptz NOT NULL DEFAULT now(),
                    updated_at timestamptz NOT NULL DEFAULT now()
                )
                """
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS chunks_document_key_idx ON {self.schema}.chunks (document_key)"
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS chunks_mayan_doc_idx ON {self.schema}.chunks (mayan_doc_id)"
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS chunks_parent_idx ON {self.schema}.chunks (parent_id)"
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS chunks_type_idx ON {self.schema}.chunks (chunk_type)"
            )
            conn.execute(
                f"CREATE INDEX IF NOT EXISTS chunks_tsv_idx ON {self.schema}.chunks USING GIN (content_tsv)"
            )
            conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
                ON {self.schema}.chunks
                USING hnsw (embedding vector_cosine_ops)
                """
            )
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.schema}.webhook_events (
                    event_key text PRIMARY KEY,
                    event_name text NOT NULL,
                    document_key text NOT NULL,
                    mayan_doc_id integer NOT NULL,
                    mayan_version_id integer,
                    status text NOT NULL CHECK (status IN ('processing', 'completed', 'failed')),
                    payload jsonb NOT NULL DEFAULT '{{}}'::jsonb,
                    error_message text,
                    created_at timestamptz NOT NULL DEFAULT now(),
                    updated_at timestamptz NOT NULL DEFAULT now()
                )
                """
            )
            conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS webhook_events_doc_idx
                ON {self.schema}.webhook_events (mayan_doc_id, mayan_version_id)
                """
            )
            # Document-level embeddings for cross-document similarity (contextual nudging)
            conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.schema}.document_embeddings (
                    document_key text PRIMARY KEY,
                    document_name text NOT NULL DEFAULT '',
                    doc_type text NOT NULL DEFAULT '',
                    mayan_doc_id integer,
                    embedding vector(384),
                    chunk_count integer NOT NULL DEFAULT 0,
                    updated_at timestamptz NOT NULL DEFAULT now()
                )
                """
            )
            conn.execute(
                f"""
                CREATE INDEX IF NOT EXISTS doc_emb_hnsw_idx
                ON {self.schema}.document_embeddings
                USING hnsw (embedding vector_cosine_ops)
                """
            )

            # Migration: add enriched_content column if missing
            conn.execute(
                f"""
                DO $$
                BEGIN
                    IF NOT EXISTS (
                        SELECT 1 FROM information_schema.columns
                        WHERE table_schema = '{self.schema}'
                          AND table_name = 'chunks'
                          AND column_name = 'enriched_content'
                    ) THEN
                        ALTER TABLE {self.schema}.chunks ADD COLUMN enriched_content text;
                    END IF;
                END $$;
                """
            )

    def replace_document(self, document_key: str, chunks: list, embeddings: dict[int, list[float]]):
        with self._connect() as conn:
            with conn.transaction():
                conn.execute(
                    f"DELETE FROM {self.schema}.chunks WHERE document_key = %s",
                    (document_key,),
                )

                for index, chunk in enumerate(chunks):
                    meta = getattr(chunk, "metadata", {}) or {}
                    embedding = embeddings.get(index)
                    enriched = meta.get("enriched_text") or chunk.text
                    conn.execute(
                        f"""
                        INSERT INTO {self.schema}.chunks (
                            id, document_key, mayan_doc_id, document_name, page_number,
                            chunk_type, parent_id, section, doc_type, start_line, end_line,
                            content, enriched_content, lines, metadata, embedding
                        ) VALUES (
                            %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s,
                            %s, %s, %s::jsonb, %s::jsonb, %s::vector
                        )
                        """,
                        (
                            str(uuid.uuid4()),
                            document_key,
                            meta.get("mayan_doc_id"),
                            meta.get("document_name", document_key),
                            meta.get("page_number", 0),
                            meta.get("chunk_type", "child"),
                            meta.get("parent_id"),
                            meta.get("section", ""),
                            meta.get("doc_type", ""),
                            meta.get("start_line", 0),
                            meta.get("end_line", 0),
                            chunk.text,
                            enriched,
                            json.dumps(getattr(chunk, "lines", []) or []),
                            json.dumps(meta),
                            _to_vector_literal(embedding) if embedding else None,
                        ),
                    )

    def vector_search(self, query_vector: list[float], limit: int = 50, document_key: str | None = None):
        filters = ["chunk_type = 'child'", "embedding IS NOT NULL"]
        vector_literal = _to_vector_literal(query_vector)
        params: list[Any] = [vector_literal]
        if document_key:
            filters.append("document_key = %s")
            params.append(document_key)

        where_clause = " AND ".join(filters)
        params.extend([vector_literal, limit])
        with self._connect() as conn:
            return conn.execute(
                f"""
                SELECT
                    id::text AS chunk_id,
                    mayan_doc_id,
                    document_key,
                    document_name,
                    page_number,
                    parent_id,
                    section,
                    start_line,
                    end_line,
                    content,
                    lines,
                    metadata,
                    1 - (embedding <=> %s::vector) AS score
                FROM {self.schema}.chunks
                WHERE {where_clause}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                params,
            ).fetchall()

    def keyword_search(self, query: str, limit: int = 50, document_key: str | None = None):
        filters = ["chunk_type = 'child'"]
        params: list[Any] = [query]
        if document_key:
            filters.append("document_key = %s")
            params.append(document_key)

        where_clause = " AND ".join(filters)
        params.append(query)
        params.append(limit)
        with self._connect() as conn:
            return conn.execute(
                f"""
                SELECT
                    id::text AS chunk_id,
                    mayan_doc_id,
                    document_key,
                    document_name,
                    page_number,
                    parent_id,
                    section,
                    start_line,
                    end_line,
                    content,
                    lines,
                    metadata,
                    ts_rank_cd(content_tsv, plainto_tsquery('english', %s)) AS score
                FROM {self.schema}.chunks
                WHERE {where_clause}
                  AND content_tsv @@ plainto_tsquery('english', %s)
                ORDER BY score DESC
                LIMIT %s
                """,
                params,
            ).fetchall()

    def get_parent_chunk(self, parent_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            return conn.execute(
                f"""
                SELECT
                    id::text AS chunk_id,
                    mayan_doc_id,
                    document_key,
                    document_name,
                    page_number,
                    parent_id,
                    section,
                    start_line,
                    end_line,
                    content,
                    lines,
                    metadata
                FROM {self.schema}.chunks
                WHERE chunk_type = 'parent' AND parent_id = %s
                LIMIT 1
                """,
                (parent_id,),
            ).fetchone()

    def get_chunks(self, chunk_ids: list[str]) -> list[dict[str, Any]]:
        if not chunk_ids:
            return []
        with self._connect() as conn:
            return conn.execute(
                f"""
                SELECT
                    id::text AS chunk_id,
                    mayan_doc_id,
                    document_key,
                    document_name,
                    page_number,
                    parent_id,
                    section,
                    start_line,
                    end_line,
                    content,
                    lines,
                    metadata
                FROM {self.schema}.chunks
                WHERE id = ANY(%s::uuid[])
                """,
                (chunk_ids,),
            ).fetchall()

    def get_document_chunks(
        self,
        document_key: str,
        limit_per_type: int = 100,
    ) -> dict[str, Any]:
        with self._connect() as conn:
            summary = conn.execute(
                f"""
                SELECT
                    document_key,
                    max(mayan_doc_id) AS mayan_doc_id,
                    max(document_name) AS document_name,
                    count(*) AS total_chunks,
                    count(*) FILTER (WHERE chunk_type = 'parent') AS parent_chunks,
                    count(*) FILTER (WHERE chunk_type = 'child') AS child_chunks,
                    min(page_number) AS first_page,
                    max(page_number) AS last_page
                FROM {self.schema}.chunks
                WHERE document_key = %s
                GROUP BY document_key
                """,
                (document_key,),
            ).fetchone()

            if not summary:
                return {}

            parents = conn.execute(
                f"""
                SELECT
                    id::text AS chunk_id,
                    page_number,
                    parent_id,
                    section,
                    start_line,
                    end_line,
                    content,
                    metadata
                FROM {self.schema}.chunks
                WHERE document_key = %s AND chunk_type = 'parent'
                ORDER BY page_number ASC, start_line ASC, chunk_id ASC
                LIMIT %s
                """,
                (document_key, limit_per_type),
            ).fetchall()

            children = conn.execute(
                f"""
                SELECT
                    id::text AS chunk_id,
                    page_number,
                    parent_id,
                    section,
                    start_line,
                    end_line,
                    content,
                    metadata
                FROM {self.schema}.chunks
                WHERE document_key = %s AND chunk_type = 'child'
                ORDER BY page_number ASC, start_line ASC, chunk_id ASC
                LIMIT %s
                """,
                (document_key, limit_per_type),
            ).fetchall()

        return {
            "summary": summary,
            "parents": parents,
            "children": children,
        }

    def has_any_chunks(self) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT EXISTS(SELECT 1 FROM {self.schema}.chunks)"
            ).fetchone()
        return bool(row["exists"]) if row else False

    def begin_webhook_event(
        self,
        event_key: str,
        event_name: str,
        document_key: str,
        mayan_doc_id: int,
        mayan_version_id: int | None,
        payload: dict[str, Any],
    ) -> tuple[bool, dict[str, Any] | None]:
        with self._connect() as conn:
            with conn.transaction():
                existing = conn.execute(
                    f"""
                    SELECT event_key, status, mayan_doc_id, mayan_version_id, updated_at
                    FROM {self.schema}.webhook_events
                    WHERE event_key = %s
                    """,
                    (event_key,),
                ).fetchone()
                if existing and existing["status"] in ("processing", "completed"):
                    return False, existing

                conn.execute(
                    f"""
                    INSERT INTO {self.schema}.webhook_events (
                        event_key, event_name, document_key, mayan_doc_id, mayan_version_id,
                        status, payload
                    )
                    VALUES (%s, %s, %s, %s, %s, 'processing', %s::jsonb)
                    ON CONFLICT (event_key) DO UPDATE
                    SET status = 'processing',
                        document_key = EXCLUDED.document_key,
                        mayan_doc_id = EXCLUDED.mayan_doc_id,
                        mayan_version_id = EXCLUDED.mayan_version_id,
                        payload = EXCLUDED.payload,
                        error_message = NULL,
                        updated_at = now()
                    """,
                    (
                        event_key,
                        event_name,
                        document_key,
                        mayan_doc_id,
                        mayan_version_id,
                        json.dumps(payload),
                    ),
                )
        return True, None

    def complete_webhook_event(self, event_key: str):
        with self._connect() as conn:
            conn.execute(
                f"""
                UPDATE {self.schema}.webhook_events
                SET status = 'completed', error_message = NULL, updated_at = now()
                WHERE event_key = %s
                """,
                (event_key,),
            )

    def fail_webhook_event(self, event_key: str, error_message: str):
        with self._connect() as conn:
            conn.execute(
                f"""
                UPDATE {self.schema}.webhook_events
                SET status = 'failed', error_message = %s, updated_at = now()
                WHERE event_key = %s
                """,
                (error_message[:2000], event_key),
            )


    # ------------------------------------------------------------------
    # Document-level embeddings (for contextual nudging)
    # ------------------------------------------------------------------

    def upsert_document_embedding(
        self,
        document_key: str,
        embedding: list[float],
        document_name: str = "",
        doc_type: str = "",
        mayan_doc_id: int | None = None,
        chunk_count: int = 0,
    ):
        """Store or update the aggregated document-level embedding."""
        with self._connect() as conn:
            conn.execute(
                f"""
                INSERT INTO {self.schema}.document_embeddings
                    (document_key, document_name, doc_type, mayan_doc_id, embedding, chunk_count, updated_at)
                VALUES (%s, %s, %s, %s, %s::vector, %s, now())
                ON CONFLICT (document_key) DO UPDATE SET
                    document_name = EXCLUDED.document_name,
                    doc_type = EXCLUDED.doc_type,
                    mayan_doc_id = EXCLUDED.mayan_doc_id,
                    embedding = EXCLUDED.embedding,
                    chunk_count = EXCLUDED.chunk_count,
                    updated_at = now()
                """,
                (document_key, document_name, doc_type, mayan_doc_id,
                 _to_vector_literal(embedding), chunk_count),
            )

    def find_similar_documents(
        self,
        query_embedding: list[float],
        exclude_document_key: str | None = None,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Find documents most similar to a given embedding."""
        vector_literal = _to_vector_literal(query_embedding)
        filters = ["embedding IS NOT NULL"]
        params: list[Any] = [vector_literal]
        if exclude_document_key:
            filters.append("document_key != %s")
            params.append(exclude_document_key)

        where_clause = " AND ".join(filters)
        params.extend([vector_literal, limit])

        with self._connect() as conn:
            return conn.execute(
                f"""
                SELECT
                    document_key,
                    document_name,
                    doc_type,
                    mayan_doc_id,
                    chunk_count,
                    1 - (embedding <=> %s::vector) AS similarity
                FROM {self.schema}.document_embeddings
                WHERE {where_clause}
                ORDER BY embedding <=> %s::vector ASC
                LIMIT %s
                """,
                params,
            ).fetchall()

    def find_similar_chunks_cross_doc(
        self,
        query_embedding: list[float],
        exclude_document_key: str,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Find child chunks from OTHER documents most similar to a given embedding."""
        vector_literal = _to_vector_literal(query_embedding)
        with self._connect() as conn:
            return conn.execute(
                f"""
                SELECT
                    id::text AS chunk_id,
                    document_key,
                    document_name,
                    mayan_doc_id,
                    page_number,
                    section,
                    content,
                    1 - (embedding <=> %s::vector) AS similarity
                FROM {self.schema}.chunks
                WHERE chunk_type = 'child'
                  AND embedding IS NOT NULL
                  AND document_key != %s
                ORDER BY embedding <=> %s::vector ASC
                LIMIT %s
                """,
                (vector_literal, exclude_document_key, vector_literal, limit),
            ).fetchall()

    def get_parent_chunks_for_documents(
        self,
        document_keys: list[str],
    ) -> list[dict[str, Any]]:
        """Get all parent chunks for a list of documents (for batch summarization)."""
        if not document_keys:
            return []
        with self._connect() as conn:
            return conn.execute(
                f"""
                SELECT
                    document_key,
                    document_name,
                    mayan_doc_id,
                    page_number,
                    section,
                    content
                FROM {self.schema}.chunks
                WHERE chunk_type = 'parent'
                  AND document_key = ANY(%s)
                ORDER BY document_key, page_number, start_line
                """,
                (document_keys,),
            ).fetchall()

    def get_document_keys_for_cabinet(self, cabinet_docs: list[int]) -> list[str]:
        """Get document_keys for a list of Mayan document IDs."""
        if not cabinet_docs:
            return []
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT DISTINCT document_key
                FROM {self.schema}.chunks
                WHERE mayan_doc_id = ANY(%s)
                """,
                (cabinet_docs,),
            ).fetchall()
            return [r["document_key"] for r in rows]

    def get_child_embeddings(self, document_key: str) -> list[list[float]]:
        """Get all child embeddings for a document (for aggregation)."""
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT embedding::text
                FROM {self.schema}.chunks
                WHERE document_key = %s AND chunk_type = 'child' AND embedding IS NOT NULL
                """,
                (document_key,),
            ).fetchall()
            result = []
            for row in rows:
                raw = row["embedding"]
                if raw:
                    values = [float(v) for v in raw.strip("[]").split(",")]
                    result.append(values)
            return result


def make_document_key(document_name: str, mayan_doc_id: int | None = None) -> str:
    if mayan_doc_id is not None:
        return f"mayan:{mayan_doc_id}"
    slug = re.sub(r"[^a-z0-9]+", "-", document_name.lower()).strip("-")
    return f"upload:{slug or 'document'}"


def _to_vector_literal(vector: list[float] | None) -> str | None:
    if vector is None:
        return None
    return "[" + ",".join(f"{float(value):.12f}" for value in vector) + "]"
