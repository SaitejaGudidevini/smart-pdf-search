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
    # Add your files:
    # "/Users/saiteja/Documents/Dev/EDMS/scripts/stage2_global_context.xlsx",
    # "/Users/saiteja/Documents/Dev/EDMS/scripts/stage2_wide_table.xlsx",
    # "/Users/saiteja/Documents/Dev/EDMS/scripts/testamazon.xlsx",
]
# ============================================================

SEP = "=" * 60

for filepath in FILES:
    print(f"\n{SEP}")
    print(f"  FILE: {filepath}")
    print(SEP)

    # --- STAGE 1: Parse (now with CellDNA extraction) ---
    dataframes, formulas, cell_dna = parser.parse(filepath)
    print(f"\n[Stage 1] Parsed {len(dataframes)} sheet(s), DNA extracted for {len(cell_dna)} sheet(s)")

    for sheet_name, df in dataframes.items():
        print(f"\n  Sheet: '{sheet_name}' — {df.shape[0]} rows x {df.shape[1]} cols")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Data:\n{df.head(5).to_string()}")

        sheet_formulas = formulas.get(sheet_name, [])
        if sheet_formulas:
            print(f"  Formulas: {sheet_formulas}")

    # --- STAGE 1.5: Column Classification + Sheet Summary (DNA-accelerated) ---
    print(f"\n[Stage 1.5] Column Classification (DNA + LLM):")
    classification_meta = enricher.classify_and_summarize(dataframes, cell_dna)
    for sheet_name, meta in classification_meta.items():
        print(f"\n  Sheet: '{sheet_name}'")
        summary = meta["summary"]
        print(f"    Data type: {summary['data_type']}")
        print(f"    Each row: {summary['entity_description']}")
        print(f"    Entity column: {summary['entity_column']}")
        print(f"    Metrics: {summary['metrics']}")
        print(f"    Dimensions: {summary['dimensions']}")
        print(f"    Temporal: {summary['temporal_columns']}")
        if summary['time_range']:
            print(f"    Time range: {summary['time_range']}")
        print(f"    DNA resolved: {meta.get('dna_resolved_count', 0)} columns")
        print(f"    Column roles:")
        for col_info in meta["classifications"]:
            print(f"      {col_info['column_name']:30s} -> {col_info['role']:12s} | {col_info['description']}")

    # --- STAGE 1.5b: Show CellDNA details ---
    if cell_dna:
        print(f"\n[CellDNA] Hidden Truth Extracted:")
        for orig_name, sheet_dna in cell_dna.items():
            print(f"\n  Sheet: '{orig_name}'")
            if sheet_dna.merged_ranges:
                print(f"    Merged ranges: {sheet_dna.merged_ranges[:5]}")
            if sheet_dna.named_tables:
                print(f"    Named tables: {sheet_dna.named_tables}")
            if sheet_dna.frozen_panes:
                print(f"    Frozen panes: {sheet_dna.frozen_panes}")
            if sheet_dna.total_rows:
                print(f"    Total rows (SUM): {sheet_dna.total_rows}")
            if sheet_dna.indent_hierarchy:
                print(f"    Indent hierarchy: detected")
            for col_name, cdna in sheet_dna.column_dna.items():
                extras = []
                if cdna.number_format != "General":
                    extras.append(f"fmt={cdna.number_format}")
                if cdna.has_total_row:
                    extras.append("totals")
                if cdna.has_formulas:
                    extras.append("formulas")
                if cdna.has_validation:
                    extras.append(f"validation={cdna.validation_values}")
                extra_str = f" ({', '.join(extras)})" if extras else ""
                print(f"      {col_name:30s} -> {cdna.semantic_type:15s} conf={cdna.type_confidence:.0%}{extra_str}")

    # --- STAGE 2a: Semantic Rows (now role-tagged) ---
    print(f"\n[Stage 2] Role-Tagged Semantic Rows:")
    for sheet_name, df in dataframes.items():
        rows = enricher.generate_semantic_rows(df, sheet_name, formulas.get(sheet_name, []))
        for r in rows[:5]:
            print(f"  {r['text']}")
        if len(rows) > 5:
            print(f"  ... ({len(rows) - 5} more rows)")

    # --- STAGE 2b: Auto Schema (now with role annotations + summary) ---
    schema = enricher.generate_all_schemas(dataframes)
    print(f"\n[Stage 2] Enriched Schema (what SQL agent sees):")
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
