"""PDF processing: text extraction, page rendering, and text highlighting."""

import fitz  # PyMuPDF
import io
import hashlib
from pathlib import Path
from dataclasses import dataclass


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
                if block["type"] != 0:  # skip non-text blocks (images, etc.)
                    continue
                for line_data in block.get("lines", []):
                    line_number += 1
                    line_text = "".join(
                        span["text"] for span in line_data["spans"]
                    ).strip()
                    if not line_text:
                        continue
                    bbox = line_data["bbox"]  # (x0, y0, x1, y1)
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
                highlight.set_colors(stroke=(1, 0.9, 0))  # yellow
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
