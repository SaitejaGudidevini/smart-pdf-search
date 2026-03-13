"""Embedding-based semantic search engine with RAG support."""

import re
import numpy as np
from fastembed import TextEmbedding
from pdf_processor import PageText


class SearchResult:
    def __init__(self, page_number: int, page_score: float, matched_lines: list,
                 page_text: str = ""):
        self.page_number = page_number
        self.page_score = page_score
        self.matched_lines = matched_lines
        self.page_text = page_text


class SearchEngine:
    def __init__(self):
        self.pages: list[PageText] = []
        self.model = None
        # Page-level embeddings
        self.page_embeddings: np.ndarray | None = None
        # Chunk-level embeddings (paragraphs/sections for finer granularity)
        self.chunk_embeddings: np.ndarray | None = None
        self.chunks: list[dict] = []  # [{page_idx, text, start_line, end_line}]

    def _load_model(self):
        if self.model is None:
            print("Loading embedding model (first time only)...")
            self.model = TextEmbedding("BAAI/bge-small-en-v1.5")
            print("Model loaded.")

    def index(self, pages: list[PageText]):
        """Build embedding index from extracted pages."""
        self._load_model()
        self.pages = pages

        # 1. Page-level embeddings
        page_texts = [p.text for p in pages]
        if page_texts:
            self.page_embeddings = np.array(
                list(self.model.embed(page_texts))
            )

        # 2. Chunk-level embeddings (split pages into ~200-word chunks)
        self.chunks = []
        chunk_texts = []
        for page_idx, page in enumerate(pages):
            page_chunks = self._split_into_chunks(page, page_idx)
            for chunk in page_chunks:
                self.chunks.append(chunk)
                chunk_texts.append(chunk["text"])

        if chunk_texts:
            self.chunk_embeddings = np.array(
                list(self.model.embed(chunk_texts))
            )

        print(f"Indexed {len(pages)} pages, {len(self.chunks)} chunks")

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Semantic search using embeddings + cosine similarity."""
        if not self.pages or self.page_embeddings is None:
            return []

        self._load_model()

        # Extract topic from query (strip intent words)
        topic = self.extract_topic(query)

        # Embed the query
        query_embedding = np.array(list(self.model.embed([topic])))[0]

        # 1. Page-level cosine similarity
        page_scores = self._cosine_similarity(query_embedding, self.page_embeddings)

        # 2. Chunk-level cosine similarity (finer granularity)
        chunk_scores = []
        if self.chunk_embeddings is not None:
            chunk_sims = self._cosine_similarity(query_embedding, self.chunk_embeddings)
            for idx, score in enumerate(chunk_sims):
                if score > 0.3:  # relevance threshold
                    chunk_scores.append((idx, float(score)))
            chunk_scores.sort(key=lambda x: x[1], reverse=True)

        # 3. Also do keyword matching for exact term hits
        keywords = self._extract_keywords(query)
        keyword_line_matches = self._keyword_search(keywords)

        # 4. Merge: combine page scores with chunk + keyword matches
        results_by_page = {}

        # Add page-level scores
        top_page_indices = np.argsort(page_scores)[::-1][:top_k * 2]
        for page_idx in top_page_indices:
            score = float(page_scores[page_idx])
            if score > 0.2:
                page_num = int(page_idx) + 1
                results_by_page[page_num] = {
                    "page_score": score,
                    "matched_lines": [],
                    "page_text": self.pages[int(page_idx)].text,
                }

        # Boost pages that have chunk-level matches
        for chunk_idx, score in chunk_scores[:20]:
            chunk = self.chunks[chunk_idx]
            page_num = chunk["page_idx"] + 1
            if page_num not in results_by_page:
                results_by_page[page_num] = {
                    "page_score": score,
                    "matched_lines": [],
                    "page_text": self.pages[chunk["page_idx"]].text,
                }
            else:
                # Boost page score by chunk relevance
                results_by_page[page_num]["page_score"] = max(
                    results_by_page[page_num]["page_score"],
                    score
                )

            # Add the matching lines from this chunk
            page = self.pages[chunk["page_idx"]]
            for line in page.lines:
                if chunk["start_line"] <= line["line_number"] <= chunk["end_line"]:
                    already = any(
                        ml["line_number"] == line["line_number"]
                        for ml in results_by_page[page_num]["matched_lines"]
                    )
                    if not already:
                        results_by_page[page_num]["matched_lines"].append({
                            **line,
                            "score": score,
                            "match_type": "semantic"
                        })

        # Add keyword matches
        for page_idx, line_data in keyword_line_matches:
            page_num = page_idx + 1
            if page_num not in results_by_page:
                results_by_page[page_num] = {
                    "page_score": 0.5,
                    "matched_lines": [],
                    "page_text": self.pages[page_idx].text,
                }
            already = any(
                ml["line_number"] == line_data["line_number"]
                for ml in results_by_page[page_num]["matched_lines"]
            )
            if not already:
                results_by_page[page_num]["matched_lines"].append({
                    **line_data,
                    "score": 0.8,
                    "match_type": "keyword"
                })
            results_by_page[page_num]["page_score"] = min(
                results_by_page[page_num]["page_score"] + 0.15, 1.0
            )

        # 5. Add context lines around each match
        for page_num, data in results_by_page.items():
            page = self.pages[page_num - 1]
            for ml in data["matched_lines"]:
                ln = ml["line_number"]
                all_lines = page.lines
                line_idx = next(
                    (i for i, l in enumerate(all_lines) if l["line_number"] == ln),
                    None
                )
                if line_idx is not None:
                    ml["context_before"] = [
                        all_lines[i]["text"]
                        for i in range(max(0, line_idx - 2), line_idx)
                    ]
                    ml["context_after"] = [
                        all_lines[i]["text"]
                        for i in range(line_idx + 1, min(len(all_lines), line_idx + 3))
                    ]

        # 6. Sort and return top_k
        sorted_results = sorted(
            results_by_page.items(),
            key=lambda x: x[1]["page_score"],
            reverse=True
        )[:top_k]

        return [
            SearchResult(
                page_number=page_num,
                page_score=data["page_score"],
                matched_lines=sorted(
                    data["matched_lines"],
                    key=lambda x: x["score"],
                    reverse=True
                )[:10],
                page_text=data["page_text"],
            )
            for page_num, data in sorted_results
        ]

    def get_page_text(self, page_number: int) -> str:
        """Get full text of a page for RAG context."""
        if 1 <= page_number <= len(self.pages):
            return self.pages[page_number - 1].text
        return ""

    def _cosine_similarity(self, query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
        """Compute cosine similarity between query and all vectors."""
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        matrix_norms = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10)
        return matrix_norms @ query_norm

    def _split_into_chunks(self, page: PageText, page_idx: int) -> list[dict]:
        """Split a page into overlapping chunks of ~150-200 words for embedding."""
        lines = page.lines
        if not lines:
            return []

        chunks = []
        chunk_size = 10  # lines per chunk
        overlap = 3      # overlapping lines

        for i in range(0, len(lines), chunk_size - overlap):
            chunk_lines = lines[i:i + chunk_size]
            if not chunk_lines:
                break
            text = " ".join(l["text"] for l in chunk_lines)
            if len(text.strip()) < 20:  # skip near-empty chunks
                continue
            chunks.append({
                "page_idx": page_idx,
                "text": text,
                "start_line": chunk_lines[0]["line_number"],
                "end_line": chunk_lines[-1]["line_number"],
            })

        # Also add the full page as one chunk for broad matching
        if page.text and len(page.text.strip()) > 20:
            chunks.append({
                "page_idx": page_idx,
                "text": page.text[:1000],  # cap to avoid embedding too-long texts
                "start_line": lines[0]["line_number"] if lines else 0,
                "end_line": lines[-1]["line_number"] if lines else 0,
            })

        return chunks

    def _keyword_search(self, keywords: list[str]) -> list[tuple[int, dict]]:
        """Direct keyword matching as fallback for exact terms."""
        results = []
        for page_idx, page in enumerate(self.pages):
            for line in page.lines:
                line_lower = line["text"].lower()
                if any(kw in line_lower for kw in keywords):
                    results.append((page_idx, line))
        return results

    def extract_topic(self, query: str) -> str:
        """Extract the actual topic from a query, stripping intent words."""
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

    def _extract_keywords(self, query: str) -> list[str]:
        """Extract meaningful keywords from query."""
        stop_words = {
            "what", "is", "the", "are", "how", "does", "do", "where",
            "when", "which", "who", "can", "could", "would", "should",
            "in", "on", "at", "to", "for", "of", "with", "from",
            "a", "an", "and", "or", "but", "this", "that", "these",
            "my", "your", "our", "their", "me", "you", "it", "its",
            "about", "tell", "find", "show", "give", "get", "has", "have",
            "i", "we", "they", "he", "she", "be", "been", "being",
            "was", "were", "will", "shall", "may", "might", "must",
            "there", "here", "also", "just", "only", "very", "much",
            "explain", "describe", "define", "summarize", "elaborate",
            "meaning", "concept", "overview", "details", "information",
        }
        topic = self.extract_topic(query)
        words = re.findall(r'\b\w+\b', topic.lower())
        keywords = [w for w in words if w not in stop_words and len(w) > 1]
        return keywords if keywords else re.findall(r'\b\w+\b', query.lower())[:3]
