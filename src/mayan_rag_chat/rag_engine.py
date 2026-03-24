"""RAG engine that runs inside Mayan EDMS — indexes documents and answers questions."""

import os
import re
import logging
from collections import Counter
from pathlib import Path

logger = logging.getLogger(__name__)

# Lazy imports — these are heavy and may not be installed at Django startup
np = None
TextEmbedding = None
QdrantClient = None
Distance = VectorParams = PointStruct = Filter = FieldCondition = MatchValue = None
BM25Okapi = None
RecursiveCharacterTextSplitter = None


def _lazy_import():
    """Import heavy ML deps only when actually needed."""
    global np, TextEmbedding, QdrantClient, Distance, VectorParams, PointStruct
    global Filter, FieldCondition, MatchValue, BM25Okapi, RecursiveCharacterTextSplitter
    if np is not None:
        return
    # Fix HuggingFace cache permissions inside Mayan container
    os.environ.setdefault('HF_HOME', '/tmp/hf_cache')
    os.environ.setdefault('FASTEMBED_CACHE_PATH', '/tmp/fastembed_cache')
    import numpy
    np = numpy
    from fastembed import TextEmbedding as _TE
    TextEmbedding = _TE
    from qdrant_client import QdrantClient as _QC
    QdrantClient = _QC
    from qdrant_client.models import (
        Distance as _D, VectorParams as _VP, PointStruct as _PS,
        Filter as _F, FieldCondition as _FC, MatchValue as _MV,
    )
    Distance, VectorParams, PointStruct = _D, _VP, _PS
    Filter, FieldCondition, MatchValue = _F, _FC, _MV
    from rank_bm25 import BM25Okapi as _BM
    BM25Okapi = _BM
    from langchain_text_splitters import RecursiveCharacterTextSplitter as _RCS
    RecursiveCharacterTextSplitter = _RCS

# Singleton engine instance
_engine = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = RAGEngine()
    return _engine


class RAGEngine:
    """All-in-one RAG engine: extraction, chunking, indexing, search, generation."""

    COLLECTION = "mayan_chunks"
    VECTOR_SIZE = 384

    def __init__(self):
        _lazy_import()
        self.client = QdrantClient(":memory:")
        self.model = None
        self.bm25_index = None
        self.bm25_chunks = []
        self.chunks = []
        self._collection_exists = False
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=512, chunk_overlap=50,
            separators=["\n\n", "\n", ". ", " "]
        )

    def _load_model(self):
        if self.model is None:
            logger.info("Loading embedding model...")
            self.model = TextEmbedding("BAAI/bge-small-en-v1.5")
            logger.info("Embedding model loaded.")

    # ------------------------------------------------------------------
    # Index a Mayan document
    # ------------------------------------------------------------------

    def index_document(self, doc_id: int, label: str, pages_text: list[dict]):
        """Index a document's pages into the RAG engine.

        Args:
            doc_id: Mayan document ID
            label: Document filename/label
            pages_text: [{"page_number": 1, "text": "..."}, ...]
        """
        self._load_model()

        # Detect doc type from text
        all_text = " ".join(p["text"] for p in pages_text)
        doc_type = self._detect_doc_type(all_text)

        # Build chunks with parent-child strategy
        new_chunks = []
        for page_data in pages_text:
            page_num = page_data["page_number"]
            text = page_data["text"]
            if not text or len(text.strip()) < 30:
                continue

            parent_id = f"doc{doc_id}_p{page_num}"

            # Parent chunk (full page, up to 4000 chars)
            new_chunks.append({
                "text": f"[{label} - Page {page_num}] {text[:4000]}",
                "metadata": {
                    "doc_id": doc_id, "label": label, "page_number": page_num,
                    "chunk_type": "parent", "parent_id": parent_id,
                    "doc_type": doc_type,
                },
            })

            # Child chunks
            child_texts = self.splitter.split_text(text)
            for child_text in child_texts:
                if len(child_text.strip()) < 30:
                    continue
                new_chunks.append({
                    "text": f"[{label} - Page {page_num}] {child_text}",
                    "metadata": {
                        "doc_id": doc_id, "label": label, "page_number": page_num,
                        "chunk_type": "child", "parent_id": parent_id,
                        "doc_type": doc_type,
                    },
                })

        if not new_chunks:
            return 0

        # Add to existing chunks (multi-document support)
        start_idx = len(self.chunks)
        self.chunks.extend(new_chunks)

        # Embed new chunks
        texts = [c["text"] for c in new_chunks]
        embeddings = list(self.model.embed(texts))

        # Create or extend Qdrant collection
        if not self._collection_exists:
            self.client.create_collection(
                collection_name=self.COLLECTION,
                vectors_config=VectorParams(size=self.VECTOR_SIZE, distance=Distance.COSINE),
            )
            self._collection_exists = True

        points = []
        for i, (chunk, emb) in enumerate(zip(new_chunks, embeddings)):
            points.append(PointStruct(
                id=start_idx + i,
                vector=emb.tolist(),
                payload={
                    "text": chunk["text"],
                    "chunk_idx": start_idx + i,
                    **chunk["metadata"],
                },
            ))

        batch_size = 100
        for i in range(0, len(points), batch_size):
            self.client.upsert(self.COLLECTION, points=points[i:i + batch_size])

        # Rebuild BM25 index (over all chunks)
        self._rebuild_bm25()

        logger.info(f"Indexed doc '{label}' (ID={doc_id}): {len(new_chunks)} chunks, type={doc_type}")
        return len(new_chunks)

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5, doc_id: int = None):
        """Hybrid search: vector + BM25 + RRF fusion.

        Returns: [{"text", "page_number", "label", "doc_id", "score", "parent_id"}]
        """
        if not self._collection_exists:
            return []

        self._load_model()
        topic = self._extract_topic(query)
        query_emb = list(self.model.embed([topic]))[0]

        # Vector search
        search_filter = Filter(must=[
            FieldCondition(key="chunk_type", match=MatchValue(value="child"))
        ])
        if doc_id:
            search_filter.must.append(
                FieldCondition(key="doc_id", match=MatchValue(value=doc_id))
            )

        vector_hits = self.client.search(
            collection_name=self.COLLECTION,
            query_vector=query_emb.tolist(),
            query_filter=search_filter,
            limit=50,
        )
        vector_results = [(hit.payload["chunk_idx"], hit.score) for hit in vector_hits]

        # BM25 search
        bm25_results = []
        if self.bm25_index and self.bm25_chunks:
            scores = self.bm25_index.get_scores(topic.lower().split())
            top_indices = np.argsort(scores)[::-1][:50]
            for idx in top_indices:
                if scores[idx] > 0:
                    chunk_idx = self.bm25_chunks[idx]["idx"]
                    if doc_id:
                        meta = self.chunks[chunk_idx]["metadata"]
                        if meta.get("doc_id") != doc_id:
                            continue
                    bm25_results.append((chunk_idx, float(scores[idx])))

        # RRF fusion
        fused_scores = {}
        k = 60
        for rank, (idx, _) in enumerate(vector_results):
            fused_scores[idx] = fused_scores.get(idx, 0) + 1.0 / (k + rank + 1)
        for rank, (idx, _) in enumerate(bm25_results):
            fused_scores[idx] = fused_scores.get(idx, 0) + 1.0 / (k + rank + 1)

        sorted_results = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for chunk_idx, score in sorted_results[:top_k * 3]:
            if chunk_idx >= len(self.chunks):
                continue
            chunk = self.chunks[chunk_idx]
            meta = chunk["metadata"]
            results.append({
                "text": chunk["text"],
                "page_number": meta.get("page_number", 0),
                "label": meta.get("label", ""),
                "doc_id": meta.get("doc_id", 0),
                "score": score,
                "parent_id": meta.get("parent_id", ""),
            })

        # Deduplicate by page
        seen = set()
        deduped = []
        for r in results:
            key = (r["doc_id"], r["page_number"])
            if key not in seen:
                seen.add(key)
                deduped.append(r)
            if len(deduped) >= top_k:
                break

        return deduped

    def get_parent_text(self, parent_id: str) -> str:
        """Get parent chunk text for RAG context."""
        if not self._collection_exists or not parent_id:
            return ""
        results = self.client.scroll(
            collection_name=self.COLLECTION,
            scroll_filter=Filter(must=[
                FieldCondition(key="chunk_type", match=MatchValue(value="parent")),
                FieldCondition(key="parent_id", match=MatchValue(value=parent_id)),
            ]),
            limit=1,
        )
        points = results[0]
        return points[0].payload.get("text", "") if points else ""

    # ------------------------------------------------------------------
    # Generate RAG answer
    # ------------------------------------------------------------------

    def generate_answer(self, query: str, results: list) -> dict | None:
        """Generate a grounded answer using retrieved chunks."""
        import httpx

        context = ""
        for r in results[:3]:
            parent_text = self.get_parent_text(r["parent_id"])
            text = parent_text or r["text"]
            context += f"\n\n--- {r['label']} PAGE {r['page_number']} ---\n{text[:2000]}"

        if not context.strip():
            return None

        system_prompt = (
            "You are a document analysis assistant. Answer based ONLY on the provided excerpts. "
            "Rules: 1) Use ONLY the provided excerpts. 2) Cite as [DocName, Page X]. "
            "3) If not enough info, say so. 4) Be concise. 5) Quote key phrases."
        )
        user_prompt = f"DOCUMENTS:{context}\n\nQUESTION: {query}"

        # Try Groq → Ollama
        api_key = os.environ.get("GROQ_API_KEY")
        if api_key:
            try:
                resp = httpx.post(
                    "https://api.groq.com/openai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                    json={
                        "model": "llama-3.3-70b-versatile",
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                        "max_tokens": 500, "temperature": 0,
                    },
                    timeout=30,
                )
                if resp.status_code == 200:
                    return {
                        "text": resp.json()["choices"][0]["message"]["content"],
                        "model": "Groq (llama-3.3-70b)",
                    }
            except Exception:
                pass

        return None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _detect_doc_type(self, text: str) -> str:
        lower = text.lower()
        academic = ["abstract", "introduction", "related work", "methodology", "conclusion"]
        if sum(1 for kw in academic if kw in lower) >= 3:
            return "research_paper"
        if re.search(r"\b(whereas|hereinafter|article\s+\d+|clause\s+\d+)\b", lower):
            return "contract"
        return "general"

    def _extract_topic(self, query: str) -> str:
        patterns = [
            r'^(?:explain|describe|what\s+is|what\s+are|tell\s+me\s+about|how\s+does)\s+(?:the\s+|a\s+)?',
        ]
        cleaned = query.strip().lower()
        for p in patterns:
            cleaned = re.sub(p, '', cleaned, flags=re.IGNORECASE)
        return cleaned.strip() or query.strip().lower()

    def _rebuild_bm25(self):
        self.bm25_chunks = []
        corpus = []
        for i, chunk in enumerate(self.chunks):
            if chunk["metadata"].get("chunk_type") == "child":
                self.bm25_chunks.append({"idx": i})
                corpus.append(chunk["text"].lower().split())
        if corpus:
            self.bm25_index = BM25Okapi(corpus)

    def get_stats(self) -> dict:
        return {
            "total_chunks": len(self.chunks),
            "parents": sum(1 for c in self.chunks if c["metadata"].get("chunk_type") == "parent"),
            "children": sum(1 for c in self.chunks if c["metadata"].get("chunk_type") == "child"),
            "unique_docs": len(set(c["metadata"].get("doc_id") for c in self.chunks)),
        }
