"""Contextual enrichment for RAG chunks.

Implements Anthropic's Contextual Retrieval approach: each child chunk gets a
context prefix that situates it within the overall document, improving embedding
quality and retrieval accuracy.

Modes
-----
- "template"  – fast, zero-cost template using doc metadata (default)
- "llm"       – LLM generates a document summary; template for per-chunk context
- "full_llm"  – LLM for both summary and per-chunk context (most accurate, slowest)
- "off"       – no enrichment
"""

import os

import httpx


class ContextEnricher:
    """Add document-level context to chunks before embedding."""

    _SUMMARY_PROMPT = (
        "Read the following text from a document and write a 2-3 sentence "
        "factual summary of what it covers and its main purpose. "
        "Be concise.\n\n"
        "TEXT:\n{text}\n\nSUMMARY:"
    )

    _CHUNK_CONTEXT_PROMPT = (
        "Document: {doc_title} ({doc_type})\n"
        "Summary: {doc_summary}\n"
        "Section: {section}\n\n"
        "Write 1-2 concise sentences explaining what this excerpt covers "
        "and how it relates to the document. Do NOT repeat the excerpt.\n\n"
        "EXCERPT:\n{chunk_text}\n\nCONTEXT:"
    )

    def __init__(self, mode: str = "template"):
        self.mode = mode
        self._summary_cache: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def enrich_chunks(
        self,
        chunks: list,
        doc_title: str,
        doc_type: str,
        full_text_preview: str = "",
    ) -> list:
        """Add ``enriched_text`` to each chunk's metadata.

        * Parent chunks: enriched_text = original text (unchanged).
        * Child chunks: enriched_text = context prefix + original text.
        """
        if self.mode == "off":
            for chunk in chunks:
                chunk.metadata["enriched_text"] = chunk.text
            return chunks

        doc_summary = self._get_summary(doc_title, doc_type, full_text_preview)

        for chunk in chunks:
            if chunk.metadata.get("chunk_type") != "child":
                chunk.metadata["enriched_text"] = chunk.text
                continue

            ctx = self._build_chunk_context(
                doc_title=doc_title,
                doc_type=doc_type,
                doc_summary=doc_summary,
                section=chunk.metadata.get("section", ""),
                page=chunk.metadata.get("page_number", 0),
                chunk_text=chunk.text,
            )
            chunk.metadata["enriched_text"] = (
                f"{ctx}\n\n{chunk.text}" if ctx else chunk.text
            )

        return chunks

    # ------------------------------------------------------------------
    # Document summary
    # ------------------------------------------------------------------

    def _get_summary(self, doc_title: str, doc_type: str, preview: str) -> str:
        cache_key = f"{doc_title}:{doc_type}"
        if cache_key in self._summary_cache:
            return self._summary_cache[cache_key]

        summary = None
        if self.mode in ("llm", "full_llm") and preview:
            summary = self._llm_call(
                self._SUMMARY_PROMPT.format(text=preview[:3000])
            )

        if not summary:
            dtype = doc_type.replace("_", " ") if doc_type else "document"
            summary = f"This is a {dtype} titled '{doc_title}'."

        self._summary_cache[cache_key] = summary
        return summary

    # ------------------------------------------------------------------
    # Per-chunk context
    # ------------------------------------------------------------------

    def _build_chunk_context(
        self,
        doc_title: str,
        doc_type: str,
        doc_summary: str,
        section: str,
        page: int,
        chunk_text: str,
    ) -> str:
        # Full-LLM mode: ask the model to situate the chunk
        if self.mode == "full_llm":
            result = self._llm_call(
                self._CHUNK_CONTEXT_PROMPT.format(
                    doc_title=doc_title,
                    doc_type=doc_type.replace("_", " "),
                    doc_summary=doc_summary,
                    section=section or "Unknown",
                    chunk_text=chunk_text[:500],
                )
            )
            if result:
                return result.strip()

        # Template context (default for "template" and "llm" modes)
        return self._template_context(
            doc_title, doc_type, doc_summary, section, page
        )

    @staticmethod
    def _template_context(
        doc_title: str,
        doc_type: str,
        doc_summary: str,
        section: str,
        page: int,
    ) -> str:
        # Spreadsheet-specific context prefix
        if doc_type == "spreadsheet":
            return ContextEnricher._tabular_context(
                doc_title, section, page, doc_summary
            )

        source = f"From '{doc_title}'"
        if doc_type and doc_type not in ("general", "general_with_images"):
            source += f" ({doc_type.replace('_', ' ')})"

        location_parts: list[str] = []
        if section and section not in ("[body]", "[image]", "[ocr]"):
            location_parts.append(f"section '{section}'")
        if page:
            location_parts.append(f"page {page}")
        location = ", ".join(location_parts)

        if location:
            return f"{source}, {location}. {doc_summary}"
        return f"{source}. {doc_summary}"

    @staticmethod
    def _tabular_context(
        doc_title: str,
        section: str,
        page: int,
        doc_summary: str,
    ) -> str:
        """Context prefix for spreadsheet chunks.

        Tells the embedding model: this text comes from a spreadsheet,
        here's the sheet name and what kind of data it contains.
        """
        source = f"From spreadsheet '{doc_title}'"
        if section and section not in ("[body]", "Columns"):
            source += f", sheet '{section}'"
        if page:
            source += f" (sheet {page})"
        return f"{source}. {doc_summary}"

    # ------------------------------------------------------------------
    # LLM helper (sync — used only at index time)
    # ------------------------------------------------------------------

    @staticmethod
    def _llm_call(prompt: str) -> str | None:
        """Try Groq → Ollama. Returns *None* on failure."""
        groq_key = os.environ.get("GROQ_API_KEY")
        if groq_key:
            try:
                with httpx.Client(timeout=15) as client:
                    resp = client.post(
                        "https://api.groq.com/openai/v1/chat/completions",
                        headers={
                            "Authorization": f"Bearer {groq_key}",
                            "Content-Type": "application/json",
                        },
                        json={
                            "model": "llama-3.3-70b-versatile",
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": 150,
                            "temperature": 0,
                        },
                    )
                    if resp.status_code == 200:
                        return resp.json()["choices"][0]["message"]["content"]
            except Exception:
                pass

        try:
            with httpx.Client(timeout=30) as client:
                resp = client.post(
                    "http://localhost:11434/api/generate",
                    json={
                        "model": "llama3.2",
                        "prompt": prompt,
                        "stream": False,
                    },
                )
                if resp.status_code == 200:
                    return resp.json().get("response")
        except Exception:
            pass

        return None
