"""Test the full Excel RAG pipeline (Stage 1 + Stage 2) with your own files.

Usage: Put your file paths below and run:
  /Users/saiteja/Documents/Dev/EDMS/src/smart-search/.venv/bin/python scripts/test_my_excel.py
"""

import sys
sys.path.insert(0, "/Users/saiteja/Documents/Dev/EDMS/src/smart-search")

from excel_parser import ExcelParser
from excel_enricher import ExcelEnricher
from chunking_pipeline import ChunkingPipeline

parser = ExcelParser()
enricher = ExcelEnricher()
pipeline = ChunkingPipeline()

# ============================================================
#  PUT YOUR FILE PATHS HERE
# ============================================================
FILES = [
    "/Users/saiteja/Documents/Dev/EDMS/src/smart-search/floating_table_test.xlsx",
    "/Users/saiteja/Documents/Dev/EDMS/src/smart-search/hidden_sheet_test.xlsx",
    "/Users/saiteja/Documents/Dev/EDMS/src/smart-search/merged_header_test.xlsx",
    # "/path/to/your/file.xlsx",
]
# ============================================================

SEP = "=" * 60

for filepath in FILES:
    print(f"\n{SEP}")
    print(f"  FILE: {filepath}")
    print(SEP)

    # --- STAGE 1: Parse ---
    dataframes, formulas = parser.parse(filepath)
    print(f"\n[Stage 1] Parsed {len(dataframes)} sheet(s)")

    for sheet_name, df in dataframes.items():
        print(f"\n  Sheet: '{sheet_name}' — {df.shape[0]} rows x {df.shape[1]} cols")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Data:\n{df.head(5).to_string()}")

        sheet_formulas = formulas.get(sheet_name, [])
        if sheet_formulas:
            print(f"  Formulas: {sheet_formulas}")

    # --- STAGE 2a: Semantic Rows (what gets embedded) ---
    print(f"\n[Stage 2] Semantic Rows:")
    for sheet_name, df in dataframes.items():
        rows = enricher.generate_semantic_rows(df, sheet_name, formulas.get(sheet_name, []))
        for r in rows[:5]:
            print(f"  {r['text']}")
        if len(rows) > 5:
            print(f"  ... ({len(rows) - 5} more rows)")

    # --- STAGE 2b: Auto Schema (what LLM sees for SQL) ---
    schema = enricher.generate_all_schemas(dataframes)
    print(f"\n[Stage 2] Auto Schema:")
    print(schema)

    # --- STAGE 2c: Chunking + Enrichment ---
    structure = parser.extract_structure(filepath)
    doc_key = filepath.split("/")[-1].replace(".", "_")
    chunks = pipeline.chunk_document(structure, filepath.split("/")[-1], document_key=doc_key)

    parents = [c for c in chunks if c.metadata.get("chunk_type") == "parent"]
    children = [c for c in chunks if c.metadata.get("chunk_type") == "child"]
    print(f"\n[Stage 2] Chunks: {len(parents)} parents, {len(children)} children")

    if children:
        print(f"\n[Stage 2] Enriched text (first child — this is what gets embedded):")
        et = children[0].metadata.get("enriched_text", children[0].text)
        print(f"  {et[:300]}")

print(f"\n{SEP}")
print("  DONE")
print(SEP)
