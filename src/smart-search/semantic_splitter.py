"""Semantic text splitter: topic-boundary detection via embedding similarity.

Replaces fixed-size character splitting with variable-size chunks that respect
semantic coherence.  Falls back to sentence-aware accumulation when no embedding
model is available.
"""

import re

import numpy as np


class SemanticSplitter:
    """Split text at semantic topic boundaries detected by embedding similarity."""

    def __init__(
        self,
        embedding_model=None,
        min_chunk_chars: int = 100,
        max_chunk_chars: int = 800,
        percentile_breakpoint: int = 25,
        window_size: int = 3,
    ):
        self._model = embedding_model
        self.min_chunk_chars = min_chunk_chars
        self.max_chunk_chars = max_chunk_chars
        self.percentile_breakpoint = percentile_breakpoint
        self.window_size = window_size

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def split_text(self, text: str) -> list[str]:
        """Return semantically coherent chunks from *text*."""
        text = (text or "").strip()
        if not text:
            return []

        sentences = self._split_sentences(text)
        if len(sentences) <= 2:
            return [text] if len(text) >= self.min_chunk_chars else []

        # Without an embedding model, fall back to sentence accumulation
        if self._model is None:
            return self._sentence_accumulate(sentences)

        windows = self._build_windows(sentences)
        embeddings = list(self._model.embed(windows))

        similarities = [
            self._cosine_sim(embeddings[i], embeddings[i + 1])
            for i in range(len(embeddings) - 1)
        ]

        breakpoints = self._detect_breakpoints(similarities)
        chunks = self._group_sentences(sentences, breakpoints)
        return self._enforce_limits(chunks)

    # ------------------------------------------------------------------
    # Sentence splitting
    # ------------------------------------------------------------------

    _SENT_RE = re.compile(
        r"(?<=[.!?])\s+(?=[A-Z\"'\(])"   # sentence-end + capital
        r"|(?<=\n)(?=\S)",                 # newline then non-space (PDF lines)
    )

    def _split_sentences(self, text: str) -> list[str]:
        parts = self._SENT_RE.split(text)
        return [p.strip() for p in parts if p and p.strip()]

    # ------------------------------------------------------------------
    # Embedding similarity
    # ------------------------------------------------------------------

    def _build_windows(self, sentences: list[str]) -> list[str]:
        """Create sentence-context windows for smoother similarity curves."""
        half = self.window_size // 2
        windows = []
        for i in range(len(sentences)):
            lo = max(0, i - half)
            hi = min(len(sentences), i + half + 1)
            windows.append(" ".join(sentences[lo:hi]))
        return windows

    @staticmethod
    def _cosine_sim(a, b) -> float:
        a, b = np.asarray(a, dtype=np.float32), np.asarray(b, dtype=np.float32)
        denom = np.linalg.norm(a) * np.linalg.norm(b)
        return float(np.dot(a, b) / denom) if denom else 0.0

    def _detect_breakpoints(self, similarities: list[float]) -> list[int]:
        """Indices where the topic shifts (similarity drops below threshold)."""
        if not similarities:
            return []
        threshold = float(np.percentile(similarities, self.percentile_breakpoint))
        return [i + 1 for i, s in enumerate(similarities) if s < threshold]

    # ------------------------------------------------------------------
    # Grouping helpers
    # ------------------------------------------------------------------

    def _group_sentences(
        self, sentences: list[str], breakpoints: list[int]
    ) -> list[str]:
        if not breakpoints:
            joined = " ".join(sentences)
            if len(joined) <= self.max_chunk_chars:
                return [joined]
            return self._sentence_accumulate(sentences)

        chunks: list[str] = []
        start = 0
        for bp in sorted(set(breakpoints)):
            chunk = " ".join(sentences[start:bp]).strip()
            if chunk:
                chunks.append(chunk)
            start = bp
        tail = " ".join(sentences[start:]).strip()
        if tail:
            chunks.append(tail)
        return chunks

    def _enforce_limits(self, chunks: list[str]) -> list[str]:
        """Merge tiny chunks with neighbours; split oversized ones."""
        # -- merge small --
        merged: list[str] = []
        buf = ""
        for c in chunks:
            if not c:
                continue
            combined = f"{buf} {c}".strip() if buf else c
            if len(combined) <= self.max_chunk_chars:
                buf = combined
            else:
                if buf:
                    merged.append(buf)
                buf = c
        if buf:
            if merged and len(buf) < self.min_chunk_chars:
                merged[-1] = f"{merged[-1]} {buf}"
            else:
                merged.append(buf)

        # -- split large --
        result: list[str] = []
        for c in merged:
            if len(c) <= self.max_chunk_chars:
                result.append(c)
            else:
                result.extend(
                    self._sentence_accumulate(self._split_sentences(c))
                )
        return [c for c in result if len(c) >= self.min_chunk_chars]

    def _sentence_accumulate(self, sentences: list[str]) -> list[str]:
        """Fallback: accumulate sentences up to *max_chunk_chars*."""
        chunks: list[str] = []
        buf = ""
        for s in sentences:
            candidate = f"{buf} {s}".strip() if buf else s
            if len(candidate) > self.max_chunk_chars and buf:
                chunks.append(buf)
                buf = s
            else:
                buf = candidate
        if buf:
            chunks.append(buf)
        return [c for c in chunks if len(c) >= self.min_chunk_chars]
