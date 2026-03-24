"""Unstructured.io-based document structure extractor.

Alternative to PyMuPDF font-heuristic heading detection. Uses unstructured's
partition_pdf with strategy='fast' for element-type classification (Title,
NarrativeText, ListItem, Table, Image, etc.) and maps results to the same
dataclass types used by the existing PDFProcessor pipeline.

Strategy for heading detection:
  - Only accept Title/Header elements as headings when they match a known
    numbered-section pattern (e.g. "1. Introduction", "1.1. Contributions",
    "A.1. Analysis") or a known structural keyword ("Abstract", "References").
  - All other Title elements are demoted to body text (level 3).
  - This avoids the common problem where table column headers, figure labels,
    and math fragments get mis-classified as headings.
"""

import re

from pdf_processor import DocumentStructure, StructuredPage, StructuredSection


# Numbered heading patterns: "1. Intro", "1.1. Contributions", "A.1.", "Appendix B"
_SECTION_NUMBER_RE = re.compile(
    r"^(?:"
    r"\d+\.\s"                     # 1. Introduction
    r"|\d+\.\d+[\.\d]*\.?\s"      # 1.1. Contributions
    r"|[A-Z]\.\s"                  # A. Appendix
    r"|[A-Z]\.\d+[\.\d]*\.?\s"    # A.1. Foo
    r"|Appendix\s+[A-Z]"          # Appendix B
    r")"
)

# Structural keywords that are always valid headings (case-insensitive match)
_MAJOR_KEYWORDS = {
    "abstract", "introduction", "conclusion", "conclusions",
    "references", "bibliography", "acknowledgments", "acknowledgements",
    "related work", "methodology", "methods", "results",
    "discussion", "appendix", "supplementary material",
}

# Patterns that indicate a "Title" element is actually noise
_NOISE_RE = re.compile(
    r"^https?://"                           # URLs
    r"|^[{<\[]"                             # Opening brackets
    r"|@[\w.-]+\.\w{2,}"                    # Email-like patterns
    r"|^[\d\s.,]+$"                         # Pure numbers/spaces
    r"|^\d+[A-Z]"                           # "1DeepSeek-AI" affiliation junk
)

# Math/symbol fragments
_MATH_SYMBOL_RE = re.compile(
    r"[\u0300-\u036f\u2000-\u2bff\U0001d400-\U0001d7ff]"
)


class UnstructuredParser:
    """Parse PDFs via unstructured and return DocumentStructure objects."""

    def __init__(self, strategy: str = "fast", languages: list[str] | None = None):
        self.strategy = strategy
        self.languages = languages or ["eng"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_structure(self, pdf_path: str) -> DocumentStructure:
        """Parse *pdf_path* and return a DocumentStructure compatible with the
        existing chunking pipeline."""
        from unstructured.partition.pdf import partition_pdf

        elements = partition_pdf(
            filename=pdf_path,
            strategy=self.strategy,
            languages=self.languages,
        )

        # Group elements by page
        pages_map: dict[int, list] = {}
        for el in elements:
            page_num = el.metadata.page_number or 1
            pages_map.setdefault(page_num, []).append(el)

        # Extract document title: first Title element on page 1 that is
        # long enough and not a section number or keyword (i.e. the paper title)
        doc_title = ""
        for el in pages_map.get(1, []):
            if el.category == "Title":
                t = (el.text or "").strip()
                if (
                    len(t) > 20
                    and not _SECTION_NUMBER_RE.match(t)
                    and t.lower() not in _MAJOR_KEYWORDS
                    and not _NOISE_RE.search(t)
                    and not _MATH_SYMBOL_RE.search(t)
                ):
                    doc_title = t
                    break

        structured_pages: list[StructuredPage] = []
        global_line = 0

        for page_num in sorted(pages_map):
            page_elements = pages_map[page_num]
            sections: list[StructuredSection] = []
            current_heading: StructuredSection | None = None
            body_lines: list[dict] = []

            for el in page_elements:
                cat = el.category
                text = (el.text or "").strip()
                if not text:
                    continue

                # Skip footers / page numbers
                if cat in ("Footer", "PageNumber", "PageBreak"):
                    continue

                global_line += 1
                bbox = self._extract_bbox(el)
                line_info = {
                    "line_number": global_line,
                    "text": text,
                    "bbox": bbox,
                    "font_size": 0,
                    "is_bold": False,
                }

                # --- Image elements ---
                if cat == "Image":
                    self._flush_body(sections, current_heading, body_lines)
                    body_lines = []
                    sections.append(StructuredSection(
                        title="[image]",
                        level=3,
                        content=f"bbox:{bbox['x0']:.1f},{bbox['y0']:.1f},{bbox['x1']:.1f},{bbox['y1']:.1f}",
                        lines=[line_info],
                        section_type="image",
                    ))
                    continue

                # --- Table elements ---
                if cat == "Table":
                    self._flush_body(sections, current_heading, body_lines)
                    body_lines = []
                    sections.append(StructuredSection(
                        title=f"[Table on page {page_num}]",
                        level=3,
                        content=text,
                        lines=[line_info],
                        section_type="table",
                    ))
                    continue

                # Classify heading level
                level = self._classify_level(cat, text)

                # --- Heading (Title / Header) ---
                if level in (1, 2):
                    self._flush_body(sections, current_heading, body_lines)
                    body_lines = []

                    current_heading = StructuredSection(
                        title=text,
                        level=level,
                        content=text,
                        lines=[line_info],
                    )
                    sections.append(current_heading)
                    continue

                # --- ListItem: accumulate into body, mark later ---
                if cat == "ListItem":
                    body_lines.append({**line_info, "_list": True})
                    continue

                # --- UncategorizedText / NarrativeText / other: body ---
                body_lines.append(line_info)

            # Flush remaining body on page
            self._flush_body(sections, current_heading, body_lines)
            structured_pages.append(StructuredPage(
                page_number=page_num,
                sections=sections,
            ))

        structure = DocumentStructure(
            title=doc_title or "Untitled",
            doc_type="general",
            pages=structured_pages,
            font_profile={"parser": "unstructured", "strategy": self.strategy},
        )
        self.detect_doc_type(structure)
        return structure

    # ------------------------------------------------------------------
    # Heading level classification
    # ------------------------------------------------------------------

    @classmethod
    def _classify_level(cls, category: str, text: str) -> int:
        """Map unstructured category + text patterns to heading levels.

        Strict policy: only numbered sections and known keywords become headings.
        Everything else is body text, even if unstructured labeled it "Title".

        Also promotes ListItem elements that match numbered-section patterns to
        headings (unstructured often classifies "1. Introduction" as ListItem).

        Returns 1 for major headings, 2 for sub-headings, 3 for body text.
        """
        stripped = text.strip()

        # ---- ListItem promotion: check if it matches a section heading ----
        # Unstructured frequently classifies "1. Introduction", "2. Math"
        # as ListItem because of the leading number+dot pattern.
        if category == "ListItem":
            if _SECTION_NUMBER_RE.match(stripped) and len(stripped) <= 50:
                # Must have some alphabetic content (not just "1. ")
                alpha_count = sum(1 for c in stripped if c.isascii() and c.isalpha())
                word_count = len(stripped.split())
                # Real section headings are short (1-8 words); longer items
                # are list items in figures/captions
                if alpha_count >= 3 and word_count <= 8:
                    if re.match(r"^\d+\.\s", stripped) and not re.match(r"^\d+\.\d+", stripped):
                        return 1
                    return 2
            return 3

        if category not in ("Title", "Header"):
            return 3

        # ---- Quick rejects for Title/Header elements ----
        if len(stripped) <= 3:
            return 3
        if len(stripped) > 80:
            return 3
        if _NOISE_RE.search(stripped):
            return 3
        # Math/symbol fragments
        if _MATH_SYMBOL_RE.search(stripped):
            return 3
        # Very few alphabetic characters
        alpha_count = sum(1 for c in stripped if c.isascii() and c.isalpha())
        if alpha_count < 3:
            return 3

        # ---- Positive matches: numbered section patterns ----
        if _SECTION_NUMBER_RE.match(stripped):
            # "1. Introduction" (single digit dot) → level 1
            if re.match(r"^\d+\.\s", stripped) and not re.match(r"^\d+\.\d+", stripped):
                return 1
            # "A. Appendix" → level 2
            return 2

        # ---- Positive matches: known structural keywords ----
        if stripped.lower() in _MAJOR_KEYWORDS:
            return 1

        # ---- Everything else: demote to body ----
        # This is intentionally strict. Table headers, figure labels, model
        # names, benchmark names, and other short text fragments are very
        # commonly mis-classified as Title by pdfminer-based extraction.
        return 3

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_bbox(el) -> dict:
        """Extract bounding box from element metadata coordinates."""
        coords = getattr(el.metadata, "coordinates", None)
        if coords and hasattr(coords, "points") and coords.points:
            pts = coords.points
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            return {
                "x0": min(xs), "y0": min(ys),
                "x1": max(xs), "y1": max(ys),
            }
        return {"x0": 0, "y0": 0, "x1": 0, "y1": 0}

    @staticmethod
    def _flush_body(
        sections: list[StructuredSection],
        current_heading: StructuredSection | None,
        body_lines: list[dict],
    ) -> None:
        """Flush accumulated body lines into section(s).

        Groups consecutive list items and marks them as section_type='list'.
        """
        if not body_lines:
            return

        parent_title = current_heading.title if current_heading else "[body]"

        # Detect whether the block is predominantly a list
        list_count = sum(1 for ln in body_lines if ln.get("_list"))
        total = len(body_lines)
        is_list = list_count >= 2 and list_count / max(total, 1) > 0.4

        # Clean _list markers from line dicts
        clean_lines = [{k: v for k, v in ln.items() if k != "_list"} for ln in body_lines]
        content = "\n".join(ln["text"] for ln in clean_lines)

        section_type = "list" if is_list else "text"
        sections.append(StructuredSection(
            title=parent_title,
            level=3,
            content=content,
            lines=clean_lines,
            section_type=section_type,
        ))

    # ------------------------------------------------------------------
    # Document-type detection (mirrors PDFProcessor.detect_doc_type)
    # ------------------------------------------------------------------

    @staticmethod
    def detect_doc_type(structure: DocumentStructure) -> str:
        """Detect document type from structure heuristics."""
        titles: list[str] = []
        all_text: list[str] = []
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
