"""Test the full chunking pipeline: extraction → chunking → enrichment.

Shows parent-child pairs with enriched text so you can verify the output.

Change the PDF_PATH below and run:
    cd /Users/saiteja/Documents/Dev/EDMS/src/smart-search
    .venv/bin/python tools/test_chunking.py
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Load GROQ_API_KEY
_env_file = Path(__file__).resolve().parent.parent.parent.parent / "docker" / ".env"
if _env_file.exists():
    load_dotenv(_env_file)

from pymupdf4llm_parser import PyMuPDF4LLMParser
from chunking_pipeline import ChunkingPipeline

# ============================================================
# CHANGE THESE LINES
# ============================================================
PDF_PATH = "/Users/saiteja/Documents/Dev/EDMS/1706.03762v7.pdf"       # <-- your PDF path
OUTPUT_MD = "/Users/saiteja/Documents/Dev/EDMS/docs/chunking_output.md"  # <-- output file name
# ============================================================

parser = PyMuPDF4LLMParser()
pipeline = ChunkingPipeline()

print(f"Processing: {PDF_PATH}")
print()

# Extract structure
structure = parser.extract_structure(PDF_PATH)
print(f"Title: {structure.title}")
print(f"Doc Type: {structure.doc_type}")
print(f"Pages: {len(structure.pages)}")
print()

# Chunk (includes enrichment)
import re
doc_key = re.sub(r"[^a-z0-9]+", "-", Path(PDF_PATH).stem.lower()).strip("-")
chunks = pipeline.chunk_document(structure, Path(PDF_PATH).stem, document_key=doc_key)

parents = [c for c in chunks if c.metadata.get("chunk_type") == "parent"]
children = [c for c in chunks if c.metadata.get("chunk_type") == "child"]

print(f"Total: {len(chunks)} chunks ({len(parents)} parents, {len(children)} children)")
print()

# Group children by parent_id
from collections import defaultdict
children_by_parent = defaultdict(list)
for c in children:
    pid = c.metadata.get("parent_id", "no-parent")
    children_by_parent[pid].append(c)

# Build markdown output
lines = [f"# Chunking Output: {structure.title}\n"]
lines.append(f"**Doc Type:** {structure.doc_type}")
lines.append(f"**Pages:** {len(structure.pages)}")
lines.append(f"**Total Chunks:** {len(chunks)} ({len(parents)} parents, {len(children)} children)")
lines.append("")
lines.append("---")
lines.append("")

for parent in parents:
    pid = parent.metadata.get("parent_id")
    section = parent.metadata.get("section", "?")
    page = parent.metadata.get("page_number", "?")
    kids = children_by_parent.get(pid, [])

    lines.append(f"## PARENT: [{section}] — page {page} — {len(parent.text)} chars")
    lines.append(f"**parent_id:** `{pid}`")
    lines.append(f"**children:** {len(kids)}")
    lines.append("")
    lines.append("```")
    lines.append(parent.text)
    lines.append("```")
    lines.append("")

    for i, child in enumerate(kids):
        enriched = child.metadata.get("enriched_text", "")
        raw = child.text

        # Split enriched into prefix + body
        prefix = ""
        if enriched.startswith("From "):
            prefix_end = enriched.find("\n\n")
            if prefix_end > 0:
                prefix = enriched[:prefix_end]

        lines.append(f"### CHILD {i+1}/{len(kids)} — {len(raw)} chars")
        if prefix:
            lines.append(f"**Enrichment:** `{prefix}`")
        lines.append("")
        lines.append("```")
        lines.append(raw)
        lines.append("```")
        lines.append("")

    lines.append("---")
    lines.append("")

with open(OUTPUT_MD, "w") as f:
    f.write("\n".join(lines))

print(f"Done — {OUTPUT_MD}")
