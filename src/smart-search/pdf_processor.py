"""PDF processing: text extraction, page rendering, and text highlighting."""

import fitz  # PyMuPDF
import io
import re
import hashlib
from pathlib import Path
from collections import Counter
from dataclasses import dataclass, field

_LIST_ITEM_RE = re.compile(
    r"^\s*(?:\d+[.):]\s|[-•*]\s|[a-z][.)]\s|\([a-z]\)\s)", re.MULTILINE
)


@dataclass
class PageText:
    page_number: int
    text: str
    lines: list  # [{line_number, text, bbox: {x0, y0, x1, y1}}]


@dataclass
class SearchHit:
    page_number: int
    line_number: int
    text: str
    bbox: dict  # {x0, y0, x1, y1}
    context_before: list[str]
    context_after: list[str]


@dataclass
class StructuredSection:
    title: str
    level: int  # 1=heading, 2=subheading, 3=body-group
    content: str
    lines: list  # [{text, bbox, font_size, is_bold}]
    section_type: str = "text"  # "text" or "image"


@dataclass
class StructuredPage:
    page_number: int
    sections: list  # list of StructuredSection


@dataclass
class DocumentStructure:
    title: str
    doc_type: str
    pages: list  # list of StructuredPage
    font_profile: dict = field(default_factory=dict)
    source_profile: dict = field(default_factory=dict)


class PDFProcessor:
    def __init__(self, cache_dir: str = "cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def extract_pages(self, pdf_path: str) -> list[PageText]:
        """Extract text from each page with line-level detail and bounding boxes."""
        doc = fitz.open(pdf_path)
        pages = []

        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]

            lines = []
            line_number = 0

            for block in blocks:
                if block["type"] != 0:
                    continue
                for line_data in block.get("lines", []):
                    line_number += 1
                    line_text = "".join(
                        span["text"] for span in line_data["spans"]
                    ).strip()
                    if not line_text:
                        continue
                    bbox = line_data["bbox"]
                    lines.append({
                        "line_number": line_number,
                        "text": line_text,
                        "bbox": {
                            "x0": bbox[0], "y0": bbox[1],
                            "x1": bbox[2], "y1": bbox[3]
                        }
                    })

            full_text = "\n".join(l["text"] for l in lines)
            pages.append(PageText(
                page_number=page_num + 1,
                text=full_text,
                lines=lines
            ))

        doc.close()
        return pages

    def apply_ocr_text(
        self,
        pages: list[PageText],
        structure: DocumentStructure,
        ocr_pages: list[dict] | None,
    ) -> tuple[list[PageText], DocumentStructure, dict]:
        """Select the best text source per page, preferring OCR when native text is weak.

        Native PDF extraction remains the default because it preserves layout, headings,
        and highlight bounding boxes. OCR is used as a fallback or override when it
        materially improves text quality for a page.
        """
        if not ocr_pages:
            structure.source_profile = {
                "mode": "native_only",
                "ocr_pages_total": 0,
                "ocr_pages_used": 0,
                "native_pages_used": len(pages),
            }
            return pages, structure, structure.source_profile

        ocr_map = {
            int(item["page_number"]): (item.get("text") or "").strip()
            for item in ocr_pages
            if item.get("page_number")
        }

        updated_pages: list[PageText] = []
        updated_structured_pages: list[StructuredPage] = []
        ocr_pages_used = 0

        for page in pages:
            native_text = (page.text or "").strip()
            ocr_text = ocr_map.get(page.page_number, "")
            use_ocr = self._should_use_ocr(native_text=native_text, ocr_text=ocr_text)

            updated_pages.append(
                PageText(
                    page_number=page.page_number,
                    text=ocr_text if use_ocr else native_text,
                    lines=page.lines,
                )
            )

            structured_page = next(
                (sp for sp in structure.pages if sp.page_number == page.page_number),
                StructuredPage(page_number=page.page_number, sections=[]),
            )

            if use_ocr:
                ocr_pages_used += 1
                updated_structured_pages.append(
                    StructuredPage(
                        page_number=page.page_number,
                        sections=self._rebuild_sections_with_ocr(structured_page.sections, ocr_text),
                    )
                )
            else:
                updated_structured_pages.append(structured_page)

        structure.pages = updated_structured_pages
        structure.source_profile = {
            "mode": "hybrid" if ocr_pages_used else "native_only",
            "ocr_pages_total": len(ocr_map),
            "ocr_pages_used": ocr_pages_used,
            "native_pages_used": len(pages) - ocr_pages_used,
        }
        # Only re-detect doc type if not already set by upstream parser (e.g., Groq)
        if not structure.doc_type or structure.doc_type == "general":
            self.detect_doc_type(structure)
        return updated_pages, structure, structure.source_profile

    def get_page_image(self, pdf_path: str, page_number: int, dpi: int = 150) -> bytes:
        """Render a PDF page as a PNG image."""
        cache_key = self._cache_key(pdf_path, page_number, dpi)
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        doc = fitz.open(pdf_path)
        page = doc[page_number - 1]
        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        doc.close()

        self._set_cached(cache_key, img_bytes)
        return img_bytes

    def get_highlighted_page_image(
        self, pdf_path: str, page_number: int,
        highlight_texts: list[str], dpi: int = 150
    ) -> bytes:
        """Render a PDF page with specific text regions highlighted in yellow."""
        doc = fitz.open(pdf_path)
        page = doc[page_number - 1]

        for text in highlight_texts:
            rects = page.search_for(text)
            for rect in rects:
                highlight = page.add_highlight_annot(rect)
                highlight.set_colors(stroke=(1, 0.9, 0))
                highlight.update()

        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        doc.close()
        return img_bytes

    def get_highlighted_page_with_bboxes(
        self, pdf_path: str, page_number: int,
        bboxes: list[dict], dpi: int = 150
    ) -> bytes:
        """Render a PDF page with specific bounding box regions highlighted."""
        doc = fitz.open(pdf_path)
        page = doc[page_number - 1]

        for bbox in bboxes:
            rect = fitz.Rect(bbox["x0"], bbox["y0"], bbox["x1"], bbox["y1"])
            highlight = page.add_highlight_annot(rect)
            highlight.set_colors(stroke=(1, 0.9, 0))
            highlight.update()

        zoom = dpi / 72
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        doc.close()
        return img_bytes

    def get_page_count(self, pdf_path: str) -> int:
        doc = fitz.open(pdf_path)
        count = len(doc)
        doc.close()
        return count

    # ------------------------------------------------------------------
    # Structure-aware extraction
    # ------------------------------------------------------------------

    def extract_structure(self, pdf_path: str) -> DocumentStructure:
        """Extract structured document layout with headings, sections, and images."""
        doc = fitz.open(pdf_path)

        # First pass: collect all font sizes to find body size
        size_counter = Counter()
        for page in doc:
            for block in page.get_text("dict")["blocks"]:
                if block["type"] != 0:
                    continue
                for line_data in block.get("lines", []):
                    for span in line_data["spans"]:
                        if span["text"].strip():
                            size_counter[round(span["size"], 1)] += len(span["text"])

        body_size = size_counter.most_common(1)[0][0] if size_counter else 12.0
        font_profile = {"body_size": body_size, "sizes_found": dict(size_counter.most_common(10))}

        # Second pass: build structured pages
        structured_pages = []
        doc_title = ""
        line_number = 0

        for page_idx in range(len(doc)):
            page = doc[page_idx]
            blocks = page.get_text("dict")["blocks"]
            sections = []
            current_heading = None
            body_lines = []

            # --- Table detection via PyMuPDF --------------------------
            table_bboxes: list[tuple] = []
            try:
                found_tables = page.find_tables()
                for tab in found_tables.tables:
                    table_bboxes.append(tab.bbox)
                    content = self._format_table(tab)
                    if content:
                        sections.append(StructuredSection(
                            title=f"[Table on page {page_idx + 1}]",
                            level=3,
                            content=content,
                            lines=[],
                            section_type="table",
                        ))
            except Exception:
                table_bboxes = []

            for block in blocks:
                if block["type"] == 1:  # image block
                    self._flush_body(sections, current_heading, body_lines)
                    body_lines = []
                    bbox = block.get("bbox", (0, 0, 0, 0))
                    sections.append(StructuredSection(
                        title="[image]", level=3,
                        content=f"bbox:{bbox[0]:.1f},{bbox[1]:.1f},{bbox[2]:.1f},{bbox[3]:.1f}",
                        lines=[], section_type="image",
                    ))
                    continue

                if block["type"] != 0:
                    continue

                # Skip text blocks that overlap with detected tables
                if table_bboxes and self._block_inside_table(block, table_bboxes):
                    continue

                for line_data in block.get("lines", []):
                    spans = line_data["spans"]
                    if not spans:
                        continue
                    line_text = "".join(s["text"] for s in spans).strip()
                    if not line_text:
                        continue

                    line_number += 1
                    max_font = max(s["size"] for s in spans)
                    is_bold = any(s["flags"] & 16 > 0 for s in spans)
                    bbox = line_data["bbox"]
                    line_info = {
                        "line_number": line_number,
                        "text": line_text,
                        "bbox": {"x0": bbox[0], "y0": bbox[1], "x1": bbox[2], "y1": bbox[3]},
                        "font_size": round(max_font, 1),
                        "is_bold": is_bold,
                    }

                    if max_font > body_size * 1.3:
                        level = 1
                    elif max_font > body_size * 1.1 and is_bold:
                        level = 2
                    elif is_bold and self._is_subsection_heading(line_text):
                        level = 2
                    else:
                        level = 3

                    if level in (1, 2):
                        self._flush_body(sections, current_heading, body_lines)
                        body_lines = []
                        current_heading = StructuredSection(
                            title=line_text, level=level,
                            content=line_text, lines=[line_info],
                        )
                        sections.append(current_heading)
                        if not doc_title and level == 1:
                            doc_title = line_text
                    else:
                        body_lines.append(line_info)

            self._flush_body(sections, current_heading, body_lines)
            structured_pages.append(StructuredPage(page_number=page_idx + 1, sections=sections))

        doc.close()

        structure = DocumentStructure(
            title=doc_title or "Untitled", doc_type="general",
            pages=structured_pages, font_profile=font_profile,
        )
        self.detect_doc_type(structure)
        return structure

    def detect_doc_type(self, structure: DocumentStructure) -> str:
        """Detect document type from structure heuristics."""
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

        academic_keywords = ["abstract", "introduction", "related work", "methodology", "conclusion", "references"]
        academic_hits = sum(1 for kw in academic_keywords if kw in combined_titles)

        if academic_hits >= 3:
            doc_type = "research_paper"
        elif re.search(r"\b(whereas|hereinafter|witnesseth)\b", combined_text, re.IGNORECASE) or \
             re.search(r"\b(article\s+\d+|clause\s+\d+)\b", combined_text, re.IGNORECASE):
            doc_type = "contract"
        else:
            doc_type = "general"

        if has_images:
            doc_type += "_with_images"

        structure.doc_type = doc_type
        return doc_type

    # Matches: "1.1.", "2.3.1", "A.1.", "1.1. Contributions", "Appendix B"
    _SUBSECTION_RE = re.compile(
        r"^(?:"
        r"\d+\.\d+[\.\d]*\.?\s"        # 1.1. or 2.3.1 followed by space
        r"|[A-Z]\.\d+\.?\s"             # A.1. followed by space
        r"|Appendix\s+[A-Z]"            # Appendix A/B/C
        r")"
    )

    @classmethod
    def _is_subsection_heading(cls, text: str) -> bool:
        """Detect subsection headings by numbering pattern (bold + pattern = heading)."""
        text = text.strip()
        # Must be a short line (headings, not body paragraphs)
        if len(text) > 80 or "\n" in text:
            return False
        return bool(cls._SUBSECTION_RE.match(text))

    @staticmethod
    def _flush_body(sections, current_heading, body_lines):
        """Flush accumulated body lines into a body-group section.

        Detects list blocks and marks them with section_type='list'.
        """
        if not body_lines:
            return
        content = "\n".join(ln["text"] for ln in body_lines)
        parent_title = current_heading.title if current_heading else "[body]"

        # Detect list blocks
        lines = content.strip().split("\n")
        list_count = sum(1 for ln in lines if _LIST_ITEM_RE.match(ln))
        section_type = "list" if list_count >= 2 and list_count / max(len(lines), 1) > 0.4 else "text"

        sections.append(StructuredSection(
            title=parent_title, level=3,
            content=content, lines=list(body_lines),
            section_type=section_type,
        ))

    @staticmethod
    def _format_table(tab) -> str:
        """Format a PyMuPDF Table object as readable pipe-delimited text."""
        try:
            rows = tab.extract()
        except Exception:
            return ""
        if not rows:
            return ""

        formatted_rows = []
        for row in rows:
            cells = [(str(cell).strip() if cell else "") for cell in row]
            formatted_rows.append(" | ".join(cells))

        if len(formatted_rows) > 1:
            # Add separator after header row
            header_line = formatted_rows[0]
            separator = "-+-".join("-" * max(len(c), 3) for c in rows[0]) if rows[0] else "---"
            return header_line + "\n" + separator + "\n" + "\n".join(formatted_rows[1:])
        return "\n".join(formatted_rows)

    @staticmethod
    def _block_inside_table(block: dict, table_bboxes: list[tuple]) -> bool:
        """Check if a text block overlaps with any detected table bbox."""
        bx0, by0, bx1, by1 = block.get("bbox", (0, 0, 0, 0))
        for tx0, ty0, tx1, ty1 in table_bboxes:
            # Check for significant overlap (>50% of block area inside table)
            ox0 = max(bx0, tx0)
            oy0 = max(by0, ty0)
            ox1 = min(bx1, tx1)
            oy1 = min(by1, ty1)
            if ox0 < ox1 and oy0 < oy1:
                overlap_area = (ox1 - ox0) * (oy1 - oy0)
                block_area = max((bx1 - bx0) * (by1 - by0), 1)
                if overlap_area / block_area > 0.5:
                    return True
        return False

    @staticmethod
    def _text_quality_score(text: str) -> tuple[int, float]:
        cleaned = (text or "").strip()
        if not cleaned:
            return (0, 0.0)
        alpha = sum(1 for ch in cleaned if ch.isalpha())
        printable = sum(1 for ch in cleaned if ch.isprintable() and not ch.isspace())
        ratio = alpha / max(printable, 1)
        return (len(cleaned), ratio)

    def _should_use_ocr(self, native_text: str, ocr_text: str) -> bool:
        if not ocr_text:
            return False

        native_len, native_ratio = self._text_quality_score(native_text)
        ocr_len, ocr_ratio = self._text_quality_score(ocr_text)

        if native_len < 80 and ocr_len >= 80:
            return True
        if native_ratio < 0.45 and ocr_ratio > native_ratio + 0.1:
            return True
        if native_len and ocr_len > native_len * 1.5 and ocr_ratio >= native_ratio:
            return True
        return False

    @staticmethod
    def _rebuild_sections_with_ocr(existing_sections: list[StructuredSection], ocr_text: str) -> list[StructuredSection]:
        if not ocr_text.strip():
            return existing_sections

        headings = [
            section for section in existing_sections
            if section.section_type != "image" and section.level in (1, 2)
        ]
        image_sections = [section for section in existing_sections if section.section_type == "image"]

        normalized = ocr_text.replace("\r\n", "\n").replace("\r", "\n")
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", normalized) if part.strip()]

        rebuilt: list[StructuredSection] = []
        rebuilt.extend(headings)
        rebuilt.extend(image_sections)

        if headings:
            title = headings[-1].title
        else:
            title = "[ocr]"

        rebuilt.append(
            StructuredSection(
                title=title,
                level=3,
                content="\n\n".join(paragraphs) if paragraphs else normalized.strip(),
                lines=[],
            )
        )
        return rebuilt

    def _cache_key(self, pdf_path: str, page_number: int, dpi: int) -> str:
        h = hashlib.md5(pdf_path.encode()).hexdigest()[:12]
        return f"{h}_p{page_number}_d{dpi}"

    def _get_cached(self, key: str) -> bytes | None:
        path = self.cache_dir / f"{key}.png"
        if path.exists():
            return path.read_bytes()
        return None

    def _set_cached(self, key: str, data: bytes):
        path = self.cache_dir / f"{key}.png"
        path.write_bytes(data)
