"""Structure extraction using pymupdf4llm — dynamic heading hierarchy from font statistics.

Replaces manual font-ratio thresholds with pymupdf4llm's full-document scan that
ranks all font sizes into a proper heading hierarchy and outputs Markdown.

Returns the same DocumentStructure interface so ChunkingPipeline works unchanged.
"""

from __future__ import annotations

import os
import re
from dataclasses import field
from collections import Counter

import httpx
import pymupdf4llm

from pdf_processor import DocumentStructure, StructuredPage, StructuredSection

# Regex to detect markdown heading lines
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$")

# Regex to detect markdown table rows
_TABLE_ROW_RE = re.compile(r"^\|.+\|$")
_TABLE_SEP_RE = re.compile(r"^\|[-:| ]+\|$")

# Regex for image placeholders emitted by pymupdf4llm
_IMAGE_RE = re.compile(r"^.*(?:picture|image).*intentionally omitted.*$", re.IGNORECASE)

# List item pattern
_LIST_ITEM_RE = re.compile(
    r"^\s*(?:\d+[.):]\s|[-•*]\s|[a-z][.)]\s|\([a-z]\)\s)", re.MULTILINE
)


class PyMuPDF4LLMParser:
    """Extract document structure via pymupdf4llm markdown conversion."""

    def extract_structure(self, pdf_path: str) -> DocumentStructure:
        """Convert PDF → Markdown → DocumentStructure."""
        # Get per-page markdown with layout metadata
        page_data = pymupdf4llm.to_markdown(
            pdf_path,
            page_chunks=True,
            show_progress=False,
        )

        structured_pages: list[StructuredPage] = []
        doc_title = ""
        all_headings: list[str] = []

        for page_info in page_data:
            meta = page_info["metadata"]
            # Handle both pymupdf4llm versions: "page_number" (>=1.x) or "page" (0.x)
            page_number = meta.get("page_number") or meta.get("page", 0)
            if isinstance(page_number, int) and page_number == 0:
                page_number = 1
            md_text = page_info["text"]
            page_boxes = page_info.get("page_boxes", [])

            sections = self._parse_markdown_page(md_text, page_number, page_boxes)
            structured_pages.append(StructuredPage(
                page_number=page_number,
                sections=sections,
            ))

            # Collect title from first H1
            if not doc_title:
                for s in sections:
                    if s.level == 1:
                        doc_title = self._clean_md_formatting(s.title)
                        break

            # Collect headings for doc type detection
            for s in sections:
                if s.level in (1, 2):
                    all_headings.append(s.title.lower())

        structure = DocumentStructure(
            title=doc_title or "Untitled",
            doc_type="general",
            pages=structured_pages,
            font_profile={"source": "pymupdf4llm"},
        )

        # Detect doc type from headings
        structure.doc_type = self._detect_doc_type(structure, all_headings)
        return structure

    # ------------------------------------------------------------------
    # Markdown → sections
    # ------------------------------------------------------------------

    def _parse_markdown_page(
        self,
        md_text: str,
        page_number: int,
        page_boxes: list[dict],
    ) -> list[StructuredSection]:
        """Parse one page of markdown into StructuredSections."""
        sections: list[StructuredSection] = []
        lines = md_text.split("\n")

        current_heading: str | None = None
        current_level: int = 3
        body_lines: list[str] = []
        line_number = 0

        # Build a box-class lookup for enrichment
        box_classes = {b.get("class", "") for b in page_boxes}

        for raw_line in lines:
            line = raw_line.rstrip()
            line_number += 1

            # Check for heading
            heading_match = _HEADING_RE.match(line)
            if heading_match:
                # Flush accumulated body
                self._flush_body(sections, current_heading, current_level,
                                 body_lines, page_number, line_number)
                body_lines = []

                hashes = heading_match.group(1)
                title_text = self._clean_md_formatting(heading_match.group(2).strip())
                level = min(len(hashes), 3)  # cap at 3 levels

                current_heading = title_text
                current_level = level

                # Add the heading itself as a section
                sections.append(StructuredSection(
                    title=title_text,
                    level=level,
                    content=title_text,
                    lines=[{"line_number": line_number, "text": title_text,
                            "bbox": {}, "font_size": 0, "is_bold": True}],
                ))
                continue

            # Check for image placeholder
            if _IMAGE_RE.match(line):
                self._flush_body(sections, current_heading, current_level,
                                 body_lines, page_number, line_number)
                body_lines = []
                # Find matching picture box for bbox
                pic_bbox = self._find_box_bbox(page_boxes, "picture")
                sections.append(StructuredSection(
                    title="[image]",
                    level=3,
                    content=f"bbox:{pic_bbox}" if pic_bbox else "image",
                    lines=[],
                    section_type="image",
                ))
                continue

            # Check for table block (accumulate consecutive table rows)
            if _TABLE_ROW_RE.match(line) or _TABLE_SEP_RE.match(line):
                # If we have non-table body lines, flush them first
                non_table = [l for l in body_lines if not _TABLE_ROW_RE.match(l) and not _TABLE_SEP_RE.match(l)]
                table_buf = [l for l in body_lines if _TABLE_ROW_RE.match(l) or _TABLE_SEP_RE.match(l)]

                if non_table and table_buf:
                    # Flush non-table body
                    self._flush_body(sections, current_heading, current_level,
                                     non_table, page_number, line_number)
                    body_lines = table_buf

                body_lines.append(line)
                continue

            # If we were accumulating table rows and this line is NOT a table row,
            # flush the table as its own section
            if body_lines and all(_TABLE_ROW_RE.match(l) or _TABLE_SEP_RE.match(l) for l in body_lines if l.strip()):
                table_content = "\n".join(body_lines)
                if table_content.strip():
                    sections.append(StructuredSection(
                        title=f"[Table on page {page_number}]",
                        level=3,
                        content=table_content.strip(),
                        lines=[],
                        section_type="table",
                    ))
                body_lines = []

            # Skip empty lines, code fences, page footers
            stripped = line.strip()
            if not stripped:
                continue
            if stripped == "```":
                continue

            # Regular body line
            body_lines.append(line)

        # Flush remaining body
        self._flush_body(sections, current_heading, current_level,
                         body_lines, page_number, line_number)

        return sections

    def _flush_body(
        self,
        sections: list[StructuredSection],
        heading: str | None,
        heading_level: int,
        body_lines: list[str],
        page_number: int,
        line_number: int,
    ) -> None:
        """Flush accumulated body lines into a section."""
        if not body_lines:
            return

        # Check if this is a table block
        non_empty = [l for l in body_lines if l.strip()]
        is_table = non_empty and all(
            _TABLE_ROW_RE.match(l) or _TABLE_SEP_RE.match(l) for l in non_empty
        )

        content = "\n".join(body_lines).strip()
        if not content:
            return

        parent_title = heading if heading else "[body]"

        if is_table:
            sections.append(StructuredSection(
                title=f"[Table on page {page_number}]",
                level=3,
                content=content,
                lines=[],
                section_type="table",
            ))
            return

        # Detect list blocks
        content_lines = content.split("\n")
        list_count = sum(1 for ln in content_lines if _LIST_ITEM_RE.match(ln))
        section_type = "list" if list_count >= 2 and list_count / max(len(content_lines), 1) > 0.4 else "text"

        # Build line metadata
        line_infos = []
        for i, text in enumerate(content_lines):
            if text.strip():
                line_infos.append({
                    "line_number": line_number - len(content_lines) + i,
                    "text": text.strip(),
                    "bbox": {},
                    "font_size": 0,
                    "is_bold": False,
                })

        sections.append(StructuredSection(
            title=parent_title,
            level=3,
            content=content,
            lines=line_infos,
            section_type=section_type,
        ))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_md_formatting(text: str) -> str:
        """Remove markdown bold/italic markers from text."""
        text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
        text = re.sub(r"_(.+?)_", r"\1", text)
        text = re.sub(r"\*(.+?)\*", r"\1", text)
        return text.strip()

    @staticmethod
    def _find_box_bbox(page_boxes: list[dict], box_class: str) -> str | None:
        """Find bounding box string for a given box class."""
        for box in page_boxes:
            if box.get("class") == box_class:
                bbox = box.get("bbox")
                if bbox:
                    return f"{bbox[0]:.1f},{bbox[1]:.1f},{bbox[2]:.1f},{bbox[3]:.1f}"
        return None

    # ------------------------------------------------------------------
    # Document type detection
    # ------------------------------------------------------------------

    # Valid doc types the chunking pipeline understands
    _VALID_DOC_TYPES = {
        "research_paper", "contract", "invoice", "resume",
        "technical_manual", "financial_report", "presentation",
        "medical_document", "general",
    }

    @classmethod
    def _detect_doc_type(cls, structure: DocumentStructure, headings: list[str]) -> str:
        """Detect document type via Groq LLM, with keyword fallback."""
        has_images = any(
            sec.section_type == "image"
            for sp in structure.pages for sec in sp.sections
        )

        # Try Groq first
        doc_type = cls._detect_doc_type_groq(structure, headings)

        # Fallback to keywords if Groq unavailable
        if not doc_type:
            doc_type = cls._detect_doc_type_keywords(headings, structure)

        if has_images:
            doc_type += "_with_images"

        structure.doc_type = doc_type
        return doc_type

    @classmethod
    def _detect_doc_type_groq(cls, structure: DocumentStructure, headings: list[str]) -> str | None:
        """Call Groq to classify document type from headings and first-page content."""
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            return None

        # Build context: headings + first 500 chars of body
        heading_list = "\n".join(f"- {h}" for h in headings[:30])
        first_page_text = ""
        for sp in structure.pages[:2]:
            for sec in sp.sections:
                if sec.section_type not in ("image", "table") and sec.level == 3:
                    first_page_text += sec.content + "\n"
                    if len(first_page_text) > 500:
                        break
            if len(first_page_text) > 500:
                break
        first_page_text = first_page_text[:500]

        valid_types = ", ".join(sorted(cls._VALID_DOC_TYPES))

        try:
            resp = httpx.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": "llama-3.3-70b-versatile",
                    "messages": [
                        {
                            "role": "system",
                            "content": (
                                "You are a document classifier. Given a document's headings and "
                                "opening text, reply with exactly one document type from this list: "
                                f"{valid_types}. Reply with ONLY the type, nothing else."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"HEADINGS:\n{heading_list}\n\n"
                                f"OPENING TEXT:\n{first_page_text}"
                            ),
                        },
                    ],
                    "max_tokens": 20,
                    "temperature": 0,
                },
                timeout=10,
            )
            if resp.status_code != 200:
                print(f"Groq doc type detection failed (HTTP {resp.status_code}), using keyword fallback")
                return None

            raw = resp.json()["choices"][0]["message"]["content"].strip().lower()
            # Clean up: strip quotes, periods, whitespace
            raw = re.sub(r"[^a-z_]", "", raw.replace(" ", "_"))

            if raw in cls._VALID_DOC_TYPES:
                print(f"Groq classified document as: {raw}")
                return raw
            else:
                print(f"Groq returned unknown type '{raw}', using keyword fallback")
                return None

        except Exception as e:
            print(f"Groq doc type detection error: {e}, using keyword fallback")
            return None

    @staticmethod
    def _detect_doc_type_keywords(headings: list[str], structure: DocumentStructure) -> str:
        """Fallback: keyword-based detection when Groq is unavailable."""
        combined = " ".join(headings)
        all_text_parts = []

        for sp in structure.pages:
            for sec in sp.sections:
                if sec.section_type != "image":
                    all_text_parts.append(sec.content)

        combined_text = " ".join(all_text_parts).lower()

        academic_keywords = [
            "abstract", "introduction", "related work",
            "methodology", "conclusion", "references",
        ]
        academic_hits = sum(1 for kw in academic_keywords if kw in combined)

        if academic_hits >= 3:
            return "research_paper"
        elif re.search(r"\b(whereas|hereinafter|witnesseth)\b", combined_text, re.IGNORECASE) or \
             re.search(r"\b(article\s+\d+|clause\s+\d+)\b", combined_text, re.IGNORECASE):
            return "contract"
        return "general"
