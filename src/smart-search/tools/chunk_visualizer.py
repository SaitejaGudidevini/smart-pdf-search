"""Chunk Coverage Visualizer — compare PDF pages against chunking output.

Upload a PDF and your chunks JSON. The tool renders each page, overlays
which text was captured by chunks (green) vs missed (red), and outputs
an interactive HTML report with coverage metrics.

Usage:
    python tools/chunk_visualizer.py <pdf_path> <chunks_json> [--out report.html]
    python tools/chunk_visualizer.py 2402.03300v3.pdf docs/deepseekmath_chunks.json

The chunks JSON must be a list of objects with at least:
    { "text": "...", "page_number": N, "chunk_type": "parent"|"child", ... }
"""

from __future__ import annotations

import argparse
import base64
import html
import io
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class LineMatch:
    line_number: int
    text: str
    bbox: dict
    matched: bool = False
    chunk_indices: list = field(default_factory=list)


@dataclass
class PageReport:
    page_number: int
    total_lines: int
    matched_lines: int
    missed_lines: int
    coverage_pct: float
    image_base64: str  # annotated page image
    missed_texts: list  # list of missed line texts
    chunk_count: int
    parent_count: int
    child_count: int


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------

def extract_page_lines(doc: fitz.Document, page_idx: int) -> list[LineMatch]:
    """Extract every text line from a PDF page with bounding boxes."""
    page = doc[page_idx]
    blocks = page.get_text("dict")["blocks"]
    lines = []
    line_num = 0

    for block in blocks:
        if block["type"] != 0:
            continue
        for line_data in block.get("lines", []):
            spans = line_data["spans"]
            if not spans:
                continue
            text = "".join(s["text"] for s in spans).strip()
            if not text:
                continue
            line_num += 1
            bbox = line_data["bbox"]
            lines.append(LineMatch(
                line_number=line_num,
                text=text,
                bbox={"x0": bbox[0], "y0": bbox[1], "x1": bbox[2], "y1": bbox[3]},
            ))
    return lines


def normalize(text: str) -> str:
    """Normalize text for fuzzy matching."""
    # Strip markdown formatting (bold, italic, code, headings)
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"_(.+?)_", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"^#{1,6}\s+", "", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Strip markdown table separators
    text = re.sub(r"\|[-:| ]+\|", "", text)
    # Strip pipe characters from table cells
    text = re.sub(r"\|", " ", text)
    # Collapse whitespace and lowercase
    text = re.sub(r"\s+", " ", text).strip().lower()
    # Remove common prefixes added by chunker
    text = re.sub(r"^\[section:\s*[^\]]*\]\s*", "", text)
    text = re.sub(r"^\[clause:\s*[^\]]*\]\s*", "", text)
    text = re.sub(r"^\[table[^\]]*\]\s*", "", text)
    text = re.sub(r"^\[figure[^\]]*\]\s*", "", text)
    text = re.sub(r"^from\s+'[^']+'\s+\([^)]+\),\s+section\s+'[^']+',\s+page\s+\d+\.\s*", "", text)
    text = re.sub(r"^this is a [^.]+\.\s*", "", text)
    # Remove superscript/subscript markers common in academic PDFs
    text = re.sub(r"\[[\d,∗†]+\]", "", text)
    return text


def match_chunks_to_lines(
    page_lines: list[LineMatch],
    page_chunks: list[dict],
) -> None:
    """Mark which PDF lines are covered by at least one chunk."""
    # Build normalized lookup of chunk texts
    chunk_texts_normalized = []
    for i, chunk in enumerate(page_chunks):
        raw = chunk.get("enriched_text") or chunk.get("text", "")
        chunk_texts_normalized.append((i, normalize(raw)))

    for line in page_lines:
        line_norm = normalize(line.text)
        if len(line_norm) < 3:
            line.matched = True  # skip trivially short lines
            continue

        for chunk_idx, chunk_norm in chunk_texts_normalized:
            # Check if the line appears in the chunk
            if line_norm in chunk_norm:
                line.matched = True
                line.chunk_indices.append(chunk_idx)
                break
            # Also try first 40 chars (handles truncated lines)
            if len(line_norm) > 15 and line_norm[:40] in chunk_norm:
                line.matched = True
                line.chunk_indices.append(chunk_idx)
                break


def render_annotated_page(
    doc: fitz.Document,
    page_idx: int,
    page_lines: list[LineMatch],
    dpi: int = 150,
) -> bytes:
    """Render a PDF page with green (matched) / red (missed) overlays."""
    page = doc[page_idx]

    for line in page_lines:
        rect = fitz.Rect(line.bbox["x0"], line.bbox["y0"],
                         line.bbox["x1"], line.bbox["y1"])
        if line.matched:
            # Green highlight — captured by a chunk
            annot = page.add_highlight_annot(rect)
            annot.set_colors(stroke=(0.2, 0.85, 0.4))
            annot.set_opacity(0.25)
            annot.update()
        else:
            # Red highlight — missed content
            annot = page.add_highlight_annot(rect)
            annot.set_colors(stroke=(0.95, 0.3, 0.3))
            annot.set_opacity(0.4)
            annot.update()

    zoom = dpi / 72
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)
    return pix.tobytes("png")


def analyze_page(
    doc: fitz.Document,
    page_idx: int,
    page_chunks: list[dict],
    dpi: int = 150,
) -> PageReport:
    """Analyze one PDF page against its chunks."""
    page_lines = extract_page_lines(doc, page_idx)
    match_chunks_to_lines(page_lines, page_chunks)

    # Render annotated image
    img_bytes = render_annotated_page(doc, page_idx, page_lines, dpi)
    img_b64 = base64.b64encode(img_bytes).decode("ascii")

    matched = sum(1 for l in page_lines if l.matched)
    missed = len(page_lines) - matched
    coverage = matched / max(len(page_lines), 1) * 100

    missed_texts = [l.text for l in page_lines if not l.matched]
    parents = sum(1 for c in page_chunks if c.get("chunk_type") == "parent")
    children = sum(1 for c in page_chunks if c.get("chunk_type") == "child")

    return PageReport(
        page_number=page_idx + 1,
        total_lines=len(page_lines),
        matched_lines=matched,
        missed_lines=missed,
        coverage_pct=coverage,
        image_base64=img_b64,
        missed_texts=missed_texts,
        chunk_count=len(page_chunks),
        parent_count=parents,
        child_count=children,
    )


# ---------------------------------------------------------------------------
# HTML report generation
# ---------------------------------------------------------------------------

def generate_html(pages: list[PageReport], pdf_name: str, chunks_name: str) -> str:
    """Generate an interactive HTML report."""
    total_lines = sum(p.total_lines for p in pages)
    total_matched = sum(p.matched_lines for p in pages)
    total_missed = sum(p.missed_lines for p in pages)
    overall_coverage = total_matched / max(total_lines, 1) * 100
    total_chunks = sum(p.chunk_count for p in pages)
    total_parents = sum(p.parent_count for p in pages)
    total_children = sum(p.child_count for p in pages)
    pages_with_gaps = sum(1 for p in pages if p.missed_lines > 0)
    zero_coverage = sum(1 for p in pages if p.coverage_pct == 0 and p.total_lines > 0)

    # Coverage color
    def cov_color(pct):
        if pct >= 95:
            return "#10b981"
        if pct >= 80:
            return "#f59e0b"
        return "#ef4444"

    # Page cards
    page_cards = []
    for p in pages:
        missed_html = ""
        if p.missed_texts:
            items = "".join(
                f'<div class="missed-line">{html.escape(t[:120])}</div>'
                for t in p.missed_texts[:15]
            )
            if len(p.missed_texts) > 15:
                items += f'<div class="missed-line" style="color:#64748b">... and {len(p.missed_texts)-15} more</div>'
            missed_html = f'<div class="missed-section"><div class="missed-title">Missed Content ({p.missed_lines} lines)</div>{items}</div>'

        page_cards.append(f"""
        <div class="page-card" id="page-{p.page_number}">
            <div class="page-header">
                <div class="page-title">Page {p.page_number}</div>
                <div class="page-stats">
                    <span class="stat-badge" style="background:{cov_color(p.coverage_pct)}20;color:{cov_color(p.coverage_pct)}">{p.coverage_pct:.0f}% coverage</span>
                    <span class="stat-badge">{p.matched_lines}/{p.total_lines} lines</span>
                    <span class="stat-badge" style="background:#3b82f620;color:#3b82f6">{p.parent_count}P {p.child_count}C</span>
                </div>
            </div>
            <div class="page-body">
                <div class="page-image">
                    <img src="data:image/png;base64,{p.image_base64}" alt="Page {p.page_number}" />
                </div>
                {missed_html}
            </div>
        </div>
        """)

    # Page nav buttons
    nav_buttons = "".join(
        f'<a href="#page-{p.page_number}" class="nav-btn" style="border-color:{cov_color(p.coverage_pct)}40;color:{cov_color(p.coverage_pct)}">{p.page_number}</a>'
        for p in pages
    )

    # Coverage histogram data
    hist_bars = ""
    for p in pages:
        height = max(p.coverage_pct * 0.8, 2)
        hist_bars += f'<div class="hist-bar" style="height:{height}px;background:{cov_color(p.coverage_pct)}" title="Page {p.page_number}: {p.coverage_pct:.0f}%"><span class="hist-label">{p.page_number}</span></div>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Chunk Coverage Report — {html.escape(pdf_name)}</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ background: #0f172a; color: #e2e8f0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; }}
.container {{ max-width: 1400px; margin: 0 auto; padding: 24px; }}

/* Header */
.header {{ text-align: center; padding: 40px 0 24px; }}
.header h1 {{ font-size: 28px; margin-bottom: 8px; }}
.header .subtitle {{ color: #64748b; font-size: 14px; }}
.file-badges {{ display: flex; gap: 12px; justify-content: center; margin-top: 16px; }}
.file-badge {{ background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 8px 16px; font-size: 13px; font-family: monospace; }}
.file-badge span {{ color: #64748b; }}

/* Summary cards */
.summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin: 24px 0; }}
.summary-card {{ background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 16px; text-align: center; }}
.summary-card .value {{ font-size: 28px; font-weight: 700; font-family: monospace; }}
.summary-card .label {{ font-size: 12px; color: #64748b; margin-top: 4px; }}

/* Coverage histogram */
.histogram {{ background: #1e293b; border: 1px solid #334155; border-radius: 12px; padding: 16px; margin: 24px 0; }}
.histogram h3 {{ font-size: 14px; color: #94a3b8; margin-bottom: 12px; }}
.hist-container {{ display: flex; align-items: flex-end; gap: 3px; height: 90px; }}
.hist-bar {{ width: 100%; min-width: 8px; border-radius: 3px 3px 0 0; position: relative; transition: opacity 0.2s; cursor: pointer; }}
.hist-bar:hover {{ opacity: 0.7; }}
.hist-label {{ position: absolute; bottom: -16px; left: 50%; transform: translateX(-50%); font-size: 9px; color: #475569; }}

/* Legend */
.legend {{ display: flex; gap: 24px; justify-content: center; margin: 16px 0; }}
.legend-item {{ display: flex; align-items: center; gap: 6px; font-size: 13px; color: #94a3b8; }}
.legend-dot {{ width: 14px; height: 14px; border-radius: 3px; }}

/* Page nav */
.page-nav {{ display: flex; gap: 4px; flex-wrap: wrap; justify-content: center; margin: 16px 0 24px; }}
.nav-btn {{ display: inline-flex; align-items: center; justify-content: center; width: 32px; height: 32px; border-radius: 6px; border: 1px solid #334155; background: #1e293b; font-size: 12px; text-decoration: none; font-weight: 600; transition: background 0.15s; }}
.nav-btn:hover {{ background: #334155; }}

/* Page cards */
.page-card {{ background: #1e293b; border: 1px solid #334155; border-radius: 12px; margin-bottom: 24px; overflow: hidden; }}
.page-header {{ display: flex; align-items: center; justify-content: space-between; padding: 12px 16px; border-bottom: 1px solid #334155; background: #1e293b; position: sticky; top: 0; z-index: 10; }}
.page-title {{ font-size: 16px; font-weight: 700; }}
.page-stats {{ display: flex; gap: 8px; }}
.stat-badge {{ font-size: 12px; padding: 3px 10px; border-radius: 6px; background: #33415520; color: #94a3b8; font-family: monospace; }}
.page-body {{ display: flex; gap: 0; }}
.page-image {{ flex: 1; padding: 8px; }}
.page-image img {{ width: 100%; border-radius: 4px; }}
.missed-section {{ width: 320px; min-width: 320px; padding: 12px; border-left: 1px solid #334155; max-height: 800px; overflow-y: auto; }}
.missed-title {{ font-size: 13px; font-weight: 700; color: #ef4444; margin-bottom: 8px; }}
.missed-line {{ font-size: 11px; color: #f87171; padding: 4px 8px; margin-bottom: 3px; background: #ef444408; border-radius: 4px; border-left: 2px solid #ef444440; font-family: monospace; word-break: break-all; }}
</style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>Chunk Coverage Report</h1>
        <div class="subtitle">Visual comparison of PDF content vs chunking output</div>
        <div class="file-badges">
            <div class="file-badge"><span>PDF:</span> {html.escape(pdf_name)}</div>
            <div class="file-badge"><span>Chunks:</span> {html.escape(chunks_name)}</div>
        </div>
    </div>

    <div class="summary">
        <div class="summary-card">
            <div class="value" style="color:{cov_color(overall_coverage)}">{overall_coverage:.1f}%</div>
            <div class="label">Overall Coverage</div>
        </div>
        <div class="summary-card">
            <div class="value" style="color:#3b82f6">{total_matched}</div>
            <div class="label">Lines Captured</div>
        </div>
        <div class="summary-card">
            <div class="value" style="color:#ef4444">{total_missed}</div>
            <div class="label">Lines Missed</div>
        </div>
        <div class="summary-card">
            <div class="value" style="color:#8b5cf6">{total_chunks}</div>
            <div class="label">Total Chunks ({total_parents}P + {total_children}C)</div>
        </div>
        <div class="summary-card">
            <div class="value">{len(pages)}</div>
            <div class="label">Pages Analyzed</div>
        </div>
        <div class="summary-card">
            <div class="value" style="color:#f59e0b">{pages_with_gaps}</div>
            <div class="label">Pages With Gaps</div>
        </div>
    </div>

    <div class="legend">
        <div class="legend-item"><div class="legend-dot" style="background:rgba(50,210,100,0.35)"></div> Captured by chunks</div>
        <div class="legend-item"><div class="legend-dot" style="background:rgba(240,70,70,0.45)"></div> Missed / not in any chunk</div>
    </div>

    <div class="histogram">
        <h3>Per-Page Coverage</h3>
        <div class="hist-container">{hist_bars}</div>
    </div>

    <div class="page-nav">{nav_buttons}</div>

    {"".join(page_cards)}
</div>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Chunk Coverage Visualizer — compare PDF pages against chunking output"
    )
    parser.add_argument("pdf", help="Path to the PDF file")
    parser.add_argument("chunks", help="Path to the chunks JSON file")
    parser.add_argument("--out", default=None, help="Output HTML file path (default: auto-named)")
    parser.add_argument("--dpi", type=int, default=150, help="Render DPI (default: 150)")
    parser.add_argument("--pages", default=None, help="Page range to analyze, e.g. '1-5' or '1,3,5'")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    chunks_path = Path(args.chunks)

    if not pdf_path.exists():
        print(f"Error: PDF not found: {pdf_path}")
        return 1
    if not chunks_path.exists():
        print(f"Error: Chunks JSON not found: {chunks_path}")
        return 1

    # Load chunks
    chunks = json.loads(chunks_path.read_text())
    print(f"Loaded {len(chunks)} chunks from {chunks_path.name}")

    # Group chunks by page number
    chunks_by_page: dict[int, list[dict]] = defaultdict(list)
    for chunk in chunks:
        pn = chunk.get("page_number")
        if pn is not None:
            chunks_by_page[int(pn)].append(chunk)

    # Open PDF
    doc = fitz.open(str(pdf_path))
    total_pages = len(doc)
    print(f"Opened {pdf_path.name}: {total_pages} pages")

    # Determine which pages to analyze
    if args.pages:
        page_indices = []
        for part in args.pages.split(","):
            part = part.strip()
            if "-" in part:
                a, b = part.split("-", 1)
                page_indices.extend(range(int(a) - 1, int(b)))
            else:
                page_indices.append(int(part) - 1)
    else:
        page_indices = list(range(total_pages))

    # Analyze each page
    reports: list[PageReport] = []
    for page_idx in page_indices:
        page_num = page_idx + 1
        page_chunks = chunks_by_page.get(page_num, [])

        # We need a fresh copy of the page for annotation (highlights are destructive)
        doc_copy = fitz.open(str(pdf_path))
        report = analyze_page(doc_copy, page_idx, page_chunks, dpi=args.dpi)
        doc_copy.close()

        status = "OK" if report.coverage_pct >= 95 else "GAPS" if report.coverage_pct >= 50 else "LOW"
        print(f"  Page {page_num:3d}: {report.coverage_pct:5.1f}% coverage ({report.matched_lines}/{report.total_lines} lines, {report.chunk_count} chunks) [{status}]")
        reports.append(report)

    doc.close()

    # Generate HTML
    out_path = args.out or f"chunk_coverage_{pdf_path.stem}_{chunks_path.stem}.html"
    html_content = generate_html(reports, pdf_path.name, chunks_path.name)
    Path(out_path).write_text(html_content)

    # Print summary
    total_lines = sum(r.total_lines for r in reports)
    total_matched = sum(r.matched_lines for r in reports)
    overall = total_matched / max(total_lines, 1) * 100
    print(f"\n{'='*60}")
    print(f"Overall coverage: {overall:.1f}% ({total_matched}/{total_lines} lines)")
    print(f"Pages with gaps:  {sum(1 for r in reports if r.missed_lines > 0)}/{len(reports)}")
    print(f"Report saved to:  {out_path}")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
