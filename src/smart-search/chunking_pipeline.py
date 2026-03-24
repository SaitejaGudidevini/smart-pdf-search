"""Chunking pipeline: structure-aware document splitting for RAG retrieval.

Improvements over naive character splitting:
1. Cross-page section merging — sections spanning page boundaries stay whole
2. Semantic splitting — child chunks split at topic boundaries, not char counts
3. Table-aware — tables are never split mid-row
4. List-aware — numbered/bulleted lists are kept as atomic units
5. Contextual enrichment — each chunk gets document-level context for embedding
"""

import re
from dataclasses import dataclass, field
from langchain_text_splitters import RecursiveCharacterTextSplitter

from semantic_splitter import SemanticSplitter
from context_enricher import ContextEnricher


@dataclass
class Chunk:
    """A chunk of text ready for embedding and vector storage."""
    text: str
    metadata: dict = field(default_factory=dict)
    lines: list = field(default_factory=list)


class ChunkingPipeline:
    """Routes documents to the right chunking strategy based on doc type."""

    CHILD_CHUNK_SIZE = 512
    CHILD_CHUNK_OVERLAP = 50
    PARENT_MAX_CHARS = 0  # 0 = no limit — store full section, truncate at retrieval time
    PARENT_MIN_CHARS = 140
    CHILD_MIN_CHARS = 90
    MERGE_SECTION_MAX_CHARS = 220
    SEPARATORS = ["\n\n", "\n", ". ", " "]

    def __init__(self, embedding_model=None, enrichment_mode="template"):
        self._child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.CHILD_CHUNK_SIZE,
            chunk_overlap=self.CHILD_CHUNK_OVERLAP,
            separators=self.SEPARATORS,
        )
        self._semantic_splitter = SemanticSplitter(
            embedding_model=embedding_model,
            min_chunk_chars=self.CHILD_MIN_CHARS,
            max_chunk_chars=self.CHILD_CHUNK_SIZE,
        )
        self._enricher = ContextEnricher(mode=enrichment_mode)
        self._use_semantic = embedding_model is not None

    def chunk_document(self, structure, document_name: str, document_key: str | None = None) -> list[Chunk]:
        """Route to the correct chunking strategy based on doc_type."""
        doc_type = structure.doc_type or "general"
        document_key = self._document_key(document_name, document_key)

        # Cross-page section merge (fix page-boundary splits)
        self._merge_cross_page_sections(structure.pages)

        if "research_paper" in doc_type:
            chunks = self._chunk_research_paper(structure, document_name, document_key)
        elif doc_type == "contract":
            chunks = self._chunk_contract(structure, document_name, document_key)
        else:
            chunks = self._chunk_general(structure, document_name, document_key)

        # Contextual enrichment (Anthropic-style context prefix)
        full_text = self._extract_preview(structure)
        self._enricher.enrich_chunks(chunks, structure.title, doc_type, full_text)

        parents = sum(1 for c in chunks if c.metadata.get("chunk_type") == "parent")
        children = len(chunks) - parents
        print(f"Chunked '{document_name}' ({doc_type}): {parents} parents, {children} children")
        return chunks

    # ------------------------------------------------------------------
    # Cross-page section merging
    # ------------------------------------------------------------------

    def _merge_cross_page_sections(self, pages: list) -> None:
        """Merge sections that span page boundaries (modifies *pages* in place)."""
        if len(pages) <= 1:
            return

        for i in range(1, len(pages)):
            prev_sections = pages[i - 1].sections
            curr_sections = pages[i].sections
            if not prev_sections or not curr_sections:
                continue

            last = prev_sections[-1]
            first = curr_sections[0]

            # Don't merge special section types
            if getattr(last, "section_type", "text") in ("image", "table"):
                continue
            if getattr(first, "section_type", "text") in ("image", "table"):
                continue

            should_merge = False
            # Body section continues on next page — only if it's truly a
            # continuation (same specific title, or content starts lowercase)
            if first.level == 3:
                if first.title == last.title and first.title not in ("[body]", ""):
                    should_merge = True
                elif first.content and first.content[0].islower():
                    should_merge = True

            if should_merge:
                last.content = self._merge_text(last.content, first.content)
                last.lines = list(last.lines) + list(first.lines)
                curr_sections.pop(0)

    # ------------------------------------------------------------------
    # Child splitting (semantic → fallback)
    # ------------------------------------------------------------------

    def _split_children(self, content: str, section_type: str = "text") -> list[str]:
        """Split content into child-sized chunks using the best available method."""
        if section_type == "list":
            return self._split_list(content)

        if self._use_semantic:
            result = self._semantic_splitter.split_text(content)
            if result:
                return result

        return self._child_splitter.split_text(content)

    _LIST_SPLIT_RE = re.compile(r"(?=\n\s*(?:\d+[.):]\s|[-•*]\s|[a-z][.)]\s))")

    def _split_list(self, content: str) -> list[str]:
        """Split a list at item boundaries — never mid-item."""
        items = self._LIST_SPLIT_RE.split(content)
        chunks: list[str] = []
        buf = ""
        for item in items:
            item = item.strip()
            if not item:
                continue
            candidate = f"{buf}\n{item}".strip() if buf else item
            if len(candidate) > self.CHILD_CHUNK_SIZE and buf:
                chunks.append(buf)
                buf = item
            else:
                buf = candidate
        if buf:
            chunks.append(buf)
        return [c for c in chunks if len(c) >= self.CHILD_MIN_CHARS]

    # ------------------------------------------------------------------
    # Research paper strategy
    # ------------------------------------------------------------------

    def _chunk_research_paper(self, structure, document_name: str, document_key: str) -> list[Chunk]:
        """Section-aware chunking for research papers."""
        chunks = []
        skip_sections = {"references", "bibliography", "acknowledgments", "acknowledgements"}

        for page in structure.pages:
            sections = self._prepare_sections(page.sections)
            for s_idx, section in enumerate(sections):
                if section.title.lower().strip() in skip_sections:
                    continue
                # Also skip if content itself starts with a references heading
                # (catches [body] sections whose content is a references block)
                first_line = (section.content or "").strip().split("\n", 1)[0].strip().lower()
                if first_line in skip_sections:
                    continue

                # Skip frontmatter: title block on page 1 that contains
                # author emails, affiliations, or "abstract" keyword in the
                # body text (indicating it's the title/author block, not content)
                if page.page_number == 1 and section.level == 1:
                    content_lower = (section.content or "").lower()
                    if self._looks_like_frontmatter(content_lower):
                        continue

                section_type = getattr(section, "section_type", "text")

                if section_type == "image":
                    chunks.append(Chunk(
                        text=f"[Figure on page {page.page_number}]: {section.content}",
                        metadata={
                            "section": section.title, "page_number": page.page_number,
                            "chunk_type": "child", "parent_id": None,
                            "doc_type": structure.doc_type, "document_name": document_name,
                            "document_key": document_key,
                        },
                        lines=section.lines,
                    ))
                    continue

                # Tables: keep as a single child chunk, never split
                if section_type == "table":
                    parent_id = f"{document_key}:p{page.page_number}_s{s_idx}"
                    chunks.append(Chunk(
                        text=f"[Table on page {page.page_number}] {section.content}",
                        metadata={
                            "section": section.title, "page_number": page.page_number,
                            "chunk_type": "child", "parent_id": parent_id,
                            "doc_type": structure.doc_type, "document_name": document_name,
                            "document_key": document_key,
                            "start_line": 0, "end_line": 0,
                        },
                        lines=section.lines,
                    ))
                    continue

                content = self._normalize_text(section.content)
                if len(content) < self.CHILD_MIN_CHARS:
                    continue

                parent_id = f"{document_key}:p{page.page_number}_s{s_idx}"
                section_prefix = self._section_prefix(section.title)
                base_meta = {
                    "section": section.title, "page_number": page.page_number,
                    "doc_type": structure.doc_type, "document_name": document_name,
                    "document_key": document_key,
                }

                # Parent chunk: full section for RAG context
                parent_body = self._parent_body(content)
                parent_text = section_prefix + parent_body
                if len(parent_body) < self.PARENT_MIN_CHARS and self._looks_like_heading_stub(parent_body):
                    continue
                chunks.append(Chunk(
                    text=parent_text,
                    metadata={
                        **base_meta, "chunk_type": "parent", "parent_id": parent_id,
                        "start_line": section.lines[0].get("line_number", 0) if section.lines else 0,
                        "end_line": section.lines[-1].get("line_number", 0) if section.lines else 0,
                    },
                    lines=section.lines,
                ))

                # Child chunks: semantic or fallback splitting
                child_texts = self._split_children(content, section_type)
                child_chunks_added = 0
                for child_text in child_texts:
                    child_text = self._normalize_text(child_text)
                    if len(child_text) < self.CHILD_MIN_CHARS:
                        continue
                    if self._is_duplicate_child(parent_body, child_text):
                        continue
                    child_lines = self._find_matching_lines(child_text, section.lines)
                    chunks.append(Chunk(
                        text=section_prefix + child_text,
                        metadata={
                            **base_meta, "chunk_type": "child", "parent_id": parent_id,
                            "start_line": child_lines[0].get("line_number", 0) if child_lines else 0,
                            "end_line": child_lines[-1].get("line_number", 0) if child_lines else 0,
                        },
                        lines=child_lines,
                    ))
                    child_chunks_added += 1

                if child_chunks_added == 0 and len(parent_body) >= self.CHILD_MIN_CHARS:
                    child_lines = self._find_matching_lines(parent_body, section.lines)
                    chunks.append(Chunk(
                        text=section_prefix + parent_body,
                        metadata={
                            **base_meta, "chunk_type": "child", "parent_id": parent_id,
                            "start_line": child_lines[0].get("line_number", 0) if child_lines else 0,
                            "end_line": child_lines[-1].get("line_number", 0) if child_lines else 0,
                        },
                        lines=child_lines,
                    ))
        return chunks

    # ------------------------------------------------------------------
    # Contract strategy
    # ------------------------------------------------------------------

    _CLAUSE_RE = re.compile(r'(?:Article\s+\d+|Clause\s+\d+|\d+\.\d+\.?\s+[A-Z])')

    def _chunk_contract(self, structure, document_name: str, document_key: str) -> list[Chunk]:
        """Clause-preserving chunking for contracts. Never split a clause."""
        chunks = []

        for page in structure.pages:
            sections = self._prepare_sections(page.sections)
            for s_idx, section in enumerate(sections):
                section_type = getattr(section, "section_type", "text")

                # Tables in contracts: keep whole
                if section_type == "table":
                    parent_id = f"{document_key}:p{page.page_number}_s{s_idx}"
                    chunks.append(Chunk(
                        text=f"[Table] {section.content}",
                        metadata={
                            "section": section.title, "page_number": page.page_number,
                            "chunk_type": "child", "parent_id": parent_id,
                            "doc_type": "contract", "document_name": document_name,
                            "document_key": document_key,
                            "start_line": 0, "end_line": 0,
                        },
                        lines=section.lines,
                    ))
                    continue

                content = self._normalize_text(section.content)
                if len(content) < self.CHILD_MIN_CHARS:
                    continue

                parent_id = f"{document_key}:p{page.page_number}_s{s_idx}"
                clause_title = section.title or f"Page {page.page_number}"
                prefix = f"[Clause: {clause_title}] "
                base_meta = {
                    "section": clause_title, "page_number": page.page_number,
                    "doc_type": "contract", "document_name": document_name,
                    "document_key": document_key,
                }

                # Parent chunk
                parent_body = self._parent_body(content)
                chunks.append(Chunk(
                    text=prefix + parent_body,
                    metadata={**base_meta, "chunk_type": "parent", "parent_id": parent_id,
                              "start_line": section.lines[0].get("line_number", 0) if section.lines else 0,
                              "end_line": section.lines[-1].get("line_number", 0) if section.lines else 0},
                    lines=section.lines,
                ))

                # If clause fits in chunk_size, keep it whole
                if self._estimate_tokens(content) <= self.CHILD_CHUNK_SIZE:
                    chunks.append(Chunk(
                        text=prefix + content,
                        metadata={**base_meta, "chunk_type": "child", "parent_id": parent_id,
                                  "start_line": section.lines[0].get("line_number", 0) if section.lines else 0,
                                  "end_line": section.lines[-1].get("line_number", 0) if section.lines else 0},
                        lines=section.lines,
                    ))
                else:
                    child_texts = self._split_children(content, section_type)
                    for child_text in child_texts:
                        child_text = self._normalize_text(child_text)
                        if len(child_text) < self.CHILD_MIN_CHARS:
                            continue
                        child_lines = self._find_matching_lines(child_text, section.lines)
                        chunks.append(Chunk(
                            text=prefix + child_text,
                            metadata={**base_meta, "chunk_type": "child", "parent_id": parent_id,
                                      "start_line": child_lines[0].get("line_number", 0) if child_lines else 0,
                                      "end_line": child_lines[-1].get("line_number", 0) if child_lines else 0},
                            lines=child_lines,
                        ))
        return chunks

    # ------------------------------------------------------------------
    # General strategy
    # ------------------------------------------------------------------

    def _chunk_general(self, structure, document_name: str, document_key: str) -> list[Chunk]:
        """General-purpose chunking with parent-child pairs."""
        chunks = []

        for page in structure.pages:
            sections = self._prepare_sections(page.sections)
            for s_idx, section in enumerate(sections):
                section_type = getattr(section, "section_type", "text")

                # Tables: single child chunk
                if section_type == "table":
                    parent_id = f"{document_key}:p{page.page_number}_s{s_idx}"
                    chunks.append(Chunk(
                        text=f"[Table] {section.content}",
                        metadata={
                            "section": section.title, "page_number": page.page_number,
                            "chunk_type": "child", "parent_id": parent_id,
                            "doc_type": "general", "document_name": document_name,
                            "document_key": document_key,
                            "start_line": 0, "end_line": 0,
                        },
                        lines=section.lines,
                    ))
                    continue

                content = self._normalize_text(section.content)
                if len(content) < self.CHILD_MIN_CHARS:
                    continue

                parent_id = f"{document_key}:p{page.page_number}_s{s_idx}"
                prefix = f"[{section.title}] " if section.title else ""
                base_meta = {
                    "section": section.title or f"Page {page.page_number}",
                    "page_number": page.page_number,
                    "doc_type": "general", "document_name": document_name,
                    "document_key": document_key,
                }

                # Parent
                parent_body = self._parent_body(content)
                chunks.append(Chunk(
                    text=prefix + parent_body,
                    metadata={**base_meta, "chunk_type": "parent", "parent_id": parent_id,
                              "start_line": section.lines[0].get("line_number", 0) if section.lines else 0,
                              "end_line": section.lines[-1].get("line_number", 0) if section.lines else 0},
                    lines=section.lines,
                ))

                # Children
                child_texts = self._split_children(content, section_type)
                for child_text in child_texts:
                    child_text = self._normalize_text(child_text)
                    if len(child_text) < self.CHILD_MIN_CHARS:
                        continue
                    child_lines = self._find_matching_lines(child_text, section.lines)
                    chunks.append(Chunk(
                        text=prefix + child_text,
                        metadata={**base_meta, "chunk_type": "child", "parent_id": parent_id,
                                  "start_line": child_lines[0].get("line_number", 0) if child_lines else 0,
                                  "end_line": child_lines[-1].get("line_number", 0) if child_lines else 0},
                        lines=child_lines,
                    ))
        return chunks

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parent_body(self, content: str) -> str:
        """Full section text for the parent chunk (no truncation at index time)."""
        if self.PARENT_MAX_CHARS > 0:
            return content[:self.PARENT_MAX_CHARS]
        return content

    @staticmethod
    def _extract_preview(structure) -> str:
        """First ~2000 chars of document text for summary generation."""
        parts: list[str] = []
        total = 0
        for page in structure.pages:
            for section in page.sections:
                if getattr(section, "section_type", "text") in ("image",):
                    continue
                content = section.content or ""
                parts.append(content)
                total += len(content)
                if total > 2000:
                    break
            if total > 2000:
                break
        return "\n\n".join(parts)[:3000]

    @staticmethod
    def _find_matching_lines(chunk_text: str, section_lines: list) -> list:
        """Find lines from the section that appear in the chunk text."""
        if not section_lines:
            return []
        matched = [ln for ln in section_lines if ln.get("text", "")[:40] in chunk_text]
        return matched if matched else section_lines[:3]

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return len(text) // 4

    def _prepare_sections(self, sections: list) -> list:
        """Merge heading stubs and tiny adjacent sections into semantic blocks."""
        prepared = []

        for section in sections:
            stype = getattr(section, "section_type", "text")
            if stype in ("image", "table"):
                prepared.append(section)
                continue

            content = self._normalize_text(section.content)
            if not content:
                continue

            if prepared and getattr(prepared[-1], "section_type", "text") not in ("image", "table"):
                previous = prepared[-1]
                previous_content = self._normalize_text(previous.content)
                same_title = (previous.title or "").strip() == (section.title or "").strip()
                heading_stub = (
                    same_title
                    and len(previous_content) <= 120
                    and previous.level in (1, 2, 3)
                )
                tiny_follow_on = (
                    len(previous_content) <= self.MERGE_SECTION_MAX_CHARS
                    or len(content) <= self.MERGE_SECTION_MAX_CHARS
                )

                # Determine if merge is safe
                can_merge = False
                if heading_stub:
                    can_merge = True
                elif tiny_follow_on:
                    # Only merge tiny sections within the same topic —
                    # never merge across different heading titles
                    prev_t = (previous.title or "").strip().lower()
                    curr_t = (section.title or "").strip().lower()
                    same_topic = (
                        same_title
                        or curr_t in ("[body]", "")
                        or prev_t in ("[body]", "")
                    )
                    # Don't merge a new heading (level 1-2) into a different section
                    if section.level in (1, 2) and not same_title:
                        same_topic = False
                    can_merge = same_topic

                if can_merge:
                    previous.content = self._merge_text(previous.content, section.content)
                    previous.lines = list(previous.lines) + list(section.lines)
                    previous.level = min(previous.level, section.level)
                    # Preserve the more specific section_type
                    if stype == "list":
                        previous.section_type = "list"
                    continue

            section.content = content
            prepared.append(section)

        return prepared

    @staticmethod
    def _merge_text(left: str, right: str) -> str:
        left = (left or "").strip()
        right = (right or "").strip()
        if not left:
            return right
        if not right:
            return left
        if left == right:
            return left
        if right.startswith(left):
            return right
        if left.endswith(right):
            return left
        return f"{left}\n{right}"

    @staticmethod
    def _normalize_text(text: str) -> str:
        text = (text or "").replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()

    @staticmethod
    def _section_prefix(title: str) -> str:
        return f"[Section: {title}] " if title else ""

    _FRONTMATTER_RE = re.compile(
        r"@[\w.-]+\.\w{2,}"          # email-like pattern
        r"|university\s+of\b"
        r"|\binstitut\w*\b"
        r"|\bdepartment\s+of\b"
        r"|\bcollege\s+of\b",
        re.IGNORECASE,
    )

    @classmethod
    def _looks_like_frontmatter(cls, text: str) -> bool:
        """Return True if text looks like a title/author/affiliation block."""
        return bool(cls._FRONTMATTER_RE.search(text))

    @staticmethod
    def _looks_like_heading_stub(text: str) -> bool:
        stripped = (text or "").strip()
        return len(stripped.split()) <= 8 and "\n" not in stripped

    @staticmethod
    def _is_duplicate_child(parent_text: str, child_text: str) -> bool:
        """True only when the child is effectively identical to the parent.

        A short section legitimately produces one parent and one child with
        the same text — the parent is for LLM context, the child for vector
        retrieval.  Only flag exact or near-exact content matches.
        """
        normalized_parent = " ".join((parent_text or "").split())
        normalized_child = " ".join((child_text or "").split())
        if not normalized_parent or not normalized_child:
            return True
        if normalized_parent == normalized_child:
            return True
        # Only flag when child is a strict substring covering >95% of parent
        if normalized_child in normalized_parent:
            return len(normalized_child) >= len(normalized_parent) * 0.95
        return False

    @staticmethod
    def _document_key(document_name: str, document_key: str | None) -> str:
        if document_key:
            return document_key
        slug = re.sub(r"[^a-z0-9]+", "-", document_name.lower()).strip("-")
        return slug or "document"
