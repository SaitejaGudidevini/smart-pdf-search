"""Hybrid search engine backed by PostgreSQL + pgvector."""

import re

from fastembed import TextEmbedding

try:
    from sentence_transformers import CrossEncoder
    _RERANKER_AVAILABLE = True
except ImportError:
    _RERANKER_AVAILABLE = False

from storage_pg import PostgresVectorStore, make_document_key


class MatchedChunk:
    """A passage from the document that matched the query."""

    def __init__(
        self,
        text: str,
        start_line: int,
        end_line: int,
        score: float,
        match_type: str,
        lines: list,
        section: str = "",
        metadata: dict = None,
    ):
        self.text = text
        self.start_line = start_line
        self.end_line = end_line
        self.score = score
        self.match_type = match_type
        self.lines = lines
        self.section = section
        self.metadata = metadata or {}


class SearchResult:
    def __init__(
        self,
        page_number: int,
        page_score: float,
        matched_chunks: list[MatchedChunk],
        page_text: str = "",
        document_id: int | None = None,
        document_name: str = "",
        document_key: str = "",
    ):
        self.page_number = page_number
        self.page_score = page_score
        self.matched_chunks = matched_chunks
        self.page_text = page_text
        self.document_id = document_id
        self.document_name = document_name
        self.document_key = document_key


class SearchEngine:
    VECTOR_SIZE = 384  # all-MiniLM-L6-v2

    def __init__(self):
        self.store = PostgresVectorStore()
        self.model = None
        self.reranker = None
        self.pages_by_document = {}

    def _load_model(self):
        if self.model is None:
            print("Loading embedding model (all-MiniLM-L6-v2)...")
            self.model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
            print("Embedding model loaded.")
        if self.reranker is None and _RERANKER_AVAILABLE:
            try:
                self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
                print("Reranker loaded.")
            except Exception:
                self.reranker = None

    def index(self, chunks: list, pages=None, document_id: int | None = None, document_name: str | None = None):
        """Index chunks into PostgreSQL + pgvector with document-scoped replacement."""
        if not chunks:
            return

        self._load_model()
        document_name = document_name or self._infer_document_name(chunks)
        document_key = make_document_key(document_name, document_id)

        if pages:
            self.pages_by_document[document_key] = pages

        child_indexes = []
        child_texts = []
        for index, chunk in enumerate(chunks):
            meta = getattr(chunk, "metadata", {}) or {}
            if meta.get("chunk_type") == "child":
                child_indexes.append(index)
                # Use enriched text for embedding (contextual retrieval)
                enriched = meta.get("enriched_text") or chunk.text
                child_texts.append(enriched)

        embeddings = {}
        if child_texts:
            for index, embedding in zip(child_indexes, self.model.embed(child_texts)):
                embeddings[index] = embedding.tolist()

        self.store.replace_document(document_key, chunks, embeddings)

        # Compute and store document-level embedding (average of child embeddings)
        if embeddings:
            import numpy as np
            emb_list = list(embeddings.values())
            doc_embedding = np.mean(emb_list, axis=0).tolist()
            first_meta = (getattr(chunks[0], "metadata", {}) or {})
            self.store.upsert_document_embedding(
                document_key=document_key,
                embedding=doc_embedding,
                document_name=document_name or "",
                doc_type=first_meta.get("doc_type", ""),
                mayan_doc_id=document_id,
                chunk_count=len(emb_list),
            )

        parent_count = sum(
            1 for chunk in chunks
            if (getattr(chunk, "metadata", {}) or {}).get("chunk_type") == "parent"
        )
        child_count = len(chunks) - parent_count
        print(f"Indexed {len(chunks)} chunks ({parent_count} parents, {child_count} children) for {document_key}")

    def search(self, query: str, top_k: int = 5, document_id: int | None = None) -> list[SearchResult]:
        """Hybrid search: pgvector similarity + PostgreSQL FTS + RRF fusion."""
        self._load_model()
        topic = self.extract_topic(query)
        query_emb = list(self.model.embed([topic]))[0].tolist()
        document_key = None
        if document_id is not None:
            document_key = make_document_key(str(document_id), document_id)

        vector_hits = self.store.vector_search(query_emb, limit=50, document_key=document_key)
        keyword_hits = self.store.keyword_search(topic, limit=50, document_key=document_key)

        vector_results = [(row["chunk_id"], float(row["score"])) for row in vector_hits]
        keyword_results = [(row["chunk_id"], float(row["score"])) for row in keyword_hits]

        fused = self._rrf_fuse(vector_results, keyword_results)
        chunk_map = {row["chunk_id"]: row for row in vector_hits + keyword_hits}

        if self.reranker and fused:
            top_to_rerank = fused[:15]
            pairs = []
            rerank_ids = []
            for chunk_id, _ in top_to_rerank:
                row = chunk_map.get(chunk_id)
                if not row:
                    continue
                pairs.append((topic, row["content"]))
                rerank_ids.append(chunk_id)
            if pairs:
                rerank_scores = self.reranker.predict(pairs)
                reranked = [(chunk_id, float(score)) for chunk_id, score in zip(rerank_ids, rerank_scores)]
                reranked.sort(key=lambda item: item[1], reverse=True)
                seen = {chunk_id for chunk_id, _ in reranked}
                fused = reranked + [item for item in fused if item[0] not in seen]

        results_by_page = {}
        for chunk_id, score in fused[:20]:
            payload = chunk_map.get(chunk_id)
            if not payload:
                continue

            page_num = payload.get("page_number", 0)
            if page_num == 0:
                continue

            doc_key = payload.get("document_key", "")
            doc_id = payload.get("mayan_doc_id")
            page_key = (doc_key, page_num)

            matched_chunk = MatchedChunk(
                text=payload["content"],
                start_line=payload.get("start_line", 0),
                end_line=payload.get("end_line", 0),
                score=score,
                match_type="hybrid",
                lines=payload.get("lines", []),
                section=payload.get("section", ""),
                metadata={
                    "parent_id": payload.get("parent_id", ""),
                    "mayan_doc_id": doc_id,
                    "document_key": doc_key,
                    "document_name": payload.get("document_name", ""),
                },
            )

            if page_key not in results_by_page:
                results_by_page[page_key] = {
                    "page_score": score,
                    "matched_chunks": [matched_chunk],
                    "page_text": self.get_page_text(page_num, doc_key),
                    "document_id": doc_id,
                    "document_name": payload.get("document_name", ""),
                    "document_key": doc_key,
                }
            else:
                existing = results_by_page[page_key]["matched_chunks"]
                overlaps = any(
                    chunk.start_line <= matched_chunk.end_line and chunk.end_line >= matched_chunk.start_line
                    for chunk in existing
                )
                if not overlaps and len(existing) < 3:
                    existing.append(matched_chunk)
                results_by_page[page_key]["page_score"] = max(results_by_page[page_key]["page_score"], score)

        sorted_results = sorted(
            results_by_page.items(),
            key=lambda item: item[1]["page_score"],
            reverse=True,
        )[:top_k]

        return [
            SearchResult(
                page_number=page_num,
                page_score=data["page_score"],
                matched_chunks=sorted(data["matched_chunks"], key=lambda chunk: chunk.score, reverse=True)[:3],
                page_text=data["page_text"],
                document_id=data["document_id"],
                document_name=data["document_name"],
                document_key=data["document_key"],
            )
            for (_, page_num), data in sorted_results
        ]

    def get_parent_chunk(self, parent_id: str) -> str | None:
        if not parent_id:
            return None
        row = self.store.get_parent_chunk(parent_id)
        return row["content"] if row else None

    def has_documents(self) -> bool:
        return self.store.has_any_chunks()

    def get_page_text(self, page_number: int, document_key: str | None = None) -> str:
        if document_key and document_key in self.pages_by_document:
            pages = self.pages_by_document[document_key]
            if 1 <= page_number <= len(pages):
                return pages[page_number - 1].text
        return ""

    def extract_topic(self, query: str) -> str:
        """Extract the actual topic, stripping intent words."""
        intent_patterns = [
            r'^(?:explain|describe|define|summarize|elaborate)\s+(?:the\s+|a\s+|an\s+)?(?:concept\s+of\s+|meaning\s+of\s+|term\s+)?',
            r'^(?:what\s+is|what\s+are|what\s+does|what\s+do)\s+(?:the\s+|a\s+|an\s+)?',
            r'^(?:tell\s+me\s+about|give\s+me\s+info\s+on|give\s+me\s+information\s+about)\s+(?:the\s+)?',
            r'^(?:how\s+does|how\s+do|how\s+is|how\s+are)\s+(?:the\s+)?',
            r'^(?:why\s+is|why\s+are|why\s+does|why\s+do)\s+(?:the\s+)?',
            r'^(?:can\s+you\s+explain|please\s+explain|could\s+you\s+explain)\s+(?:the\s+|a\s+)?',
            r'^(?:show\s+me|find)\s+(?:the\s+|all\s+)?(?:information\s+about\s+|details\s+about\s+|info\s+on\s+)?',
        ]
        cleaned = query.strip().lower()
        for pattern in intent_patterns:
            cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
        return cleaned.strip() if cleaned.strip() else query.strip().lower()

    def _rrf_fuse(self, vector_results: list, keyword_results: list, k: int = 60) -> list:
        scores = {}
        for rank, (chunk_id, _) in enumerate(vector_results):
            scores[chunk_id] = scores.get(chunk_id, 0) + 1.0 / (k + rank + 1)
        for rank, (chunk_id, _) in enumerate(keyword_results):
            scores[chunk_id] = scores.get(chunk_id, 0) + 1.0 / (k + rank + 1)
        return sorted(scores.items(), key=lambda item: item[1], reverse=True)

    @staticmethod
    def _infer_document_name(chunks: list) -> str:
        for chunk in chunks:
            meta = getattr(chunk, "metadata", {}) or {}
            name = meta.get("document_name")
            if name:
                return name
        return "document"
