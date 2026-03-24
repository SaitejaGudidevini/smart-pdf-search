"""Extract PDF structure to a readable Markdown file.

Change the variables below and run:
    cd /Users/saiteja/Documents/Dev/EDMS/src/smart-search
    .venv/bin/python tools/extract_to_markdown.py
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Load GROQ_API_KEY from docker/.env for LLM-based doc type detection
_env_file = Path(__file__).resolve().parent.parent.parent.parent / "docker" / ".env"
if _env_file.exists():
    load_dotenv(_env_file)

from pymupdf4llm_parser import PyMuPDF4LLMParser

# ============================================================
# CHANGE THESE TWO LINES
# ============================================================
PDF_PATH = "/Users/saiteja/Documents/Dev/EDMS/2402.03300v3.pdf"       # <-- your PDF path
OUTPUT_MD = "/Users/saiteja/Documents/Dev/EDMS/docs/extraction_transformer_imaeges.md"    # <-- output file name
# ============================================================

parser = PyMuPDF4LLMParser()
structure = parser.extract_structure(PDF_PATH)

lines = [f"# Extraction Report: {structure.title}\n"]
lines.append(f"**Doc Type:** {structure.doc_type}")
lines.append(f"**Pages:** {len(structure.pages)}")
total = sum(len(p.sections) for p in structure.pages)
lines.append(f"**Total Sections:** {total}\n---\n")

for page in structure.pages:
    lines.append(f"## Page {page.page_number}\n")
    if not page.sections:
        lines.append("*(no sections extracted)*\n")
        continue
    for s in page.sections:
        label = "H1" if s.level == 1 else "H2" if s.level == 2 else "BODY"
        if s.section_type == "table": label = "TABLE"
        elif s.section_type == "image": label = "IMAGE"
        elif s.section_type == "list": label = "LIST"
        lines.append(f"### [{label}] {s.title} — {len(s.content)} chars\n")
        lines.append(f"```\n{s.content}\n```\n")
    lines.append("---\n")

with open(OUTPUT_MD, "w") as f:
    f.write("\n".join(lines))

print(f"Done — {OUTPUT_MD}")
