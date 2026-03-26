"""Test Stage 2: See how your DataFrames become semantic rows + schema.

Usage:
  1. Put your file paths in FILES below
  2. Run: /Users/saiteja/Documents/Dev/EDMS/src/smart-search/.venv/bin/python scripts/test_stage2.py
"""

import sys
sys.path.insert(0, "/Users/saiteja/Documents/Dev/EDMS/src/smart-search")

from excel_parser import ExcelParser
from excel_enricher import ExcelEnricher

parser = ExcelParser()
enricher = ExcelEnricher()

# ============================================================
#  PUT YOUR FILE PATHS HERE
# ============================================================
FILES = [
    "/Users/saiteja/Downloads/804dd615-4626-4a11-911e-a3d535fdbf50.xlsx"
]
# ============================================================

for filepath in FILES:
    print("=" * 60)
    print(f"FILE: {filepath}")
    print("=" * 60)

    # Stage 1 output (your DataFrames)
    dataframes, formulas = parser.parse(filepath)

    for sheet_name, df in dataframes.items():
        print(f"\n--- STAGE 1 OUTPUT: DataFrame '{sheet_name}' ---")
        print(df.to_string())

        # Stage 2 output: semantic rows
        rows = enricher.generate_semantic_rows(df, sheet_name, formulas.get(sheet_name, []))
        print(f"\n--- STAGE 2 OUTPUT: Semantic Rows ---")
        for r in rows:
            print(f"  {r['text']}")

    # Stage 2 output: schema
    print(f"\n--- STAGE 2 OUTPUT: Schema (what LLM sees) ---")
    print(enricher.generate_all_schemas(dataframes))
    print()
