"""Docling-based document structure extractor.

Uses IBM's Docling library for layout-aware PDF parsing, mapping output
to the same dataclass types used by pdf_processor.py so the chunking
pipeline works without changes.
"""

import re
from collections import defaultdict

from docling.document_converter import DocumentConverter
from docling_core.types.doc import DocItemLabel
from docling_core.types.doc.document import (
    SectionHeaderItem,
    TextItem,
    TableItem,
    ListItem,
    PictureItem,
)

from pdf_processor import DocumentStructure, StructuredPage, StructuredSection


class DoclingParser:
    """Extract document structure using Docling's deep-learning layout model."""

    # Patterns for classifying heading depth from numbering:
    # Level-1 (top sections): "1.", "2.", "A." or no number at all (Abstract, References)
    _LEVEL1_RE = re.compile(
        r"^(?:\d+\.\s|[A-Z]\.\s|Abstract|Introduction|Conclusion|References|Appendix)"
    )
    # Level-2 (subsections): "1.1.", "2.3.", "A.1." etc.
    _SUBSECTION_RE = re.compile(
        r"^(?:"
        r"\d+\.\d+[\.\d]*\.?\s"       # 1.1. or 2.3.1 followed by space
        r"|[A-Z]\.\d+[\.\d]*\.?\s"    # A.1. or A.1.1. followed by space
        r"|Appendix\s+[A-Z]"          # Appendix A/B/C
        r")"
    )

    def __init__(self):
        self._converter = DocumentConverter()

    def extract_structure(self, pdf_path: str) -> DocumentStructure:
        """Parse a PDF and return a DocumentStructure compatible with ChunkingPipeline."""
        result = self._converter.convert(pdf_path)
        doc = result.document

        # Collect items grouped by page
        page_items: dict[int, list] = defaultdict(list)
        line_counter = 0

        for item, level in doc.iterate_items():
            # Determine page number from provenance
            page_no = self._get_page_no(item)
            if page_no is None:
                continue

            bbox = self._get_bbox(item)
            line_counter += 1

            entry = {
                "item": item,
                "level": level,
                "page_no": page_no,
                "bbox": bbox,
                "line_number": line_counter,
            }
            page_items[page_no].append(entry)

        # Build structured pages
        structured_pages = []
        doc_title = ""

        for page_no in sorted(page_items.keys()):
            sections = []
            current_heading = None
            body_lines = []

            for entry in page_items[page_no]:
                item = entry["item"]
                bbox = entry["bbox"]
                line_num = entry["line_number"]

                if isinstance(item, SectionHeaderItem):
                    # Flush accumulated body text
                    self._flush_body(sections, current_heading, body_lines)
                    body_lines = []

                    heading_text = item.text.strip()
                    heading_level = self._classify_heading_level(
                        heading_text, item.level
                    )

                    line_info = {
                        "line_number": line_num,
                        "text": heading_text,
                        "bbox": bbox,
                    }

                    current_heading = StructuredSection(
                        title=heading_text,
                        level=heading_level,
                        content=heading_text,
                        lines=[line_info],
                    )
                    sections.append(current_heading)

                    if not doc_title and heading_level == 1:
                        doc_title = heading_text

                elif isinstance(item, TableItem):
                    self._flush_body(sections, current_heading, body_lines)
                    body_lines = []

                    # Export table as markdown text
                    try:
                        table_text = item.export_to_markdown(doc)
                    except Exception:
                        table_text = "[table]"

                    sections.append(StructuredSection(
                        title=f"[Table on page {page_no}]",
                        level=3,
                        content=table_text,
                        lines=[],
                        section_type="table",
                    ))

                elif isinstance(item, PictureItem):
                    self._flush_body(sections, current_heading, body_lines)
                    body_lines = []

                    bbox_str = (
                        f"bbox:{bbox['x0']:.1f},{bbox['y0']:.1f},"
                        f"{bbox['x1']:.1f},{bbox['y1']:.1f}"
                        if bbox else "bbox:0,0,0,0"
                    )
                    sections.append(StructuredSection(
                        title="[image]",
                        level=3,
                        content=bbox_str,
                        lines=[],
                        section_type="image",
                    ))

                elif isinstance(item, ListItem):
                    text = item.text.strip()
                    if not text:
                        continue
                    marker = getattr(item, "marker", "-") or "-"
                    line_info = {
                        "line_number": line_num,
                        "text": f"{marker} {text}",
                        "bbox": bbox,
                    }
                    body_lines.append(line_info)

                elif isinstance(item, TextItem):
                    label = item.label
                    text = item.text.strip()
                    if not text:
                        continue

                    # Skip page headers/footers and references
                    if label in (
                        DocItemLabel.PAGE_HEADER,
                        DocItemLabel.PAGE_FOOTER,
                        DocItemLabel.REFERENCE,
                    ):
                        continue

                    # Title items (Docling marks the document title)
                    if label == DocItemLabel.TITLE:
                        self._flush_body(sections, current_heading, body_lines)
                        body_lines = []

                        line_info = {
                            "line_number": line_num,
                            "text": text,
                            "bbox": bbox,
                        }
                        current_heading = StructuredSection(
                            title=text,
                            level=1,
                            content=text,
                            lines=[line_info],
                        )
                        sections.append(current_heading)
                        if not doc_title:
                            doc_title = text
                        continue

                    # Regular paragraph / caption / footnote / text
                    line_info = {
                        "line_number": line_num,
                        "text": text,
                        "bbox": bbox,
                    }
                    body_lines.append(line_info)

            # Flush remaining body text on this page
            self._flush_body(sections, current_heading, body_lines)
            structured_pages.append(StructuredPage(
                page_number=page_no,
                sections=sections,
            ))

        structure = DocumentStructure(
            title=doc_title or "Untitled",
            doc_type="general",
            pages=structured_pages,
            font_profile={"parser": "docling"},
        )
        self.detect_doc_type(structure)
        return structure

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @classmethod
    def _classify_heading_level(cls, text: str, docling_level: int) -> int:
        """Determine heading level (1 or 2) from text numbering and Docling depth.

        Docling often reports a flat tree (all level-1).  We use the section
        numbering pattern to infer the real hierarchy:
        - "1. Introduction" -> level 1  (single-number top sections)
        - "1.1. Contributions" -> level 2  (multi-number subsections)
        - "Abstract", "References" -> level 1  (well-known top sections)
        """
        text = text.strip()
        # If Docling already distinguishes depth, honour it
        if docling_level > 1:
            return 2

        # Use numbering patterns to differentiate
        if cls._SUBSECTION_RE.match(text):
            return 2
        return 1

    @staticmethod
    def _get_page_no(item) -> int | None:
        """Extract page number from item provenance (1-based)."""
        prov = getattr(item, "prov", None)
        if prov and len(prov) > 0:
            return prov[0].page_no
        return None

    @staticmethod
    def _get_bbox(item) -> dict:
        """Extract bounding box as {x0, y0, x1, y1} dict."""
        prov = getattr(item, "prov", None)
        if prov and len(prov) > 0:
            bb = prov[0].bbox
            return {"x0": bb.l, "y0": bb.t, "x1": bb.r, "y1": bb.b}
        return {"x0": 0, "y0": 0, "x1": 0, "y1": 0}

    @staticmethod
    def _flush_body(sections, current_heading, body_lines):
        """Flush accumulated body lines into a body-group section."""
        if not body_lines:
            return
        content = "\n".join(ln["text"] for ln in body_lines)
        parent_title = current_heading.title if current_heading else "[body]"

        # Detect list blocks
        list_re = re.compile(
            r"^\s*(?:\d+[.):]\s|[-\u2022*]\s|[a-z][.)]\s|\([a-z]\)\s)",
            re.MULTILINE,
        )
        lines = content.strip().split("\n")
        list_count = sum(1 for ln in lines if list_re.match(ln))
        section_type = (
            "list"
            if list_count >= 2 and list_count / max(len(lines), 1) > 0.4
            else "text"
        )

        sections.append(StructuredSection(
            title=parent_title,
            level=3,
            content=content,
            lines=list(body_lines),
            section_type=section_type,
        ))

    @staticmethod
    def detect_doc_type(structure: DocumentStructure) -> str:
        """Detect document type from structure heuristics (same logic as PDFProcessor)."""
        titles = []
        all_text = []
        has_images = False

        for sp in structure.pages:
            for sec in sp.sections:
                if sec.section_type == "image":
                    has_images = True
                    continue
                if sec.level in (1, 2):
                    titles.append(sec.title.lower())
                all_text.append(sec.content)

        combined_titles = " ".join(titles)
        combined_text = " ".join(all_text)

        academic_keywords = [
            "abstract", "introduction", "related work",
            "methodology", "conclusion", "references",
        ]
        academic_hits = sum(1 for kw in academic_keywords if kw in combined_titles)

        if academic_hits >= 3:
            doc_type = "research_paper"
        elif re.search(
            r"\b(whereas|hereinafter|witnesseth)\b", combined_text, re.IGNORECASE
        ) or re.search(
            r"\b(article\s+\d+|clause\s+\d+)\b", combined_text, re.IGNORECASE
        ):
            doc_type = "contract"
        else:
            doc_type = "general"

        if has_images:
            doc_type += "_with_images"

        structure.doc_type = doc_type
        return doc_type
