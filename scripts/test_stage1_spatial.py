"""Test the spatial Excel parser (rebuilt Stage 1).

Shows how each sheet is split into islands, classified, and mapped to DocumentStructure.

Usage:
  1. Put your file paths in FILES below
  2. Run: /Users/saiteja/Documents/Dev/EDMS/src/smart-search/.venv/bin/python scripts/test_stage1_spatial.py
"""

import sys
sys.path.insert(0, "/Users/saiteja/Documents/Dev/EDMS/src/smart-search")

from excel_parser import ExcelParser

parser = ExcelParser()

# ============================================================
#  PUT YOUR FILE PATHS HERE
# ============================================================
FILES = [
    "/Users/saiteja/Documents/Dev/EDMS/src/smart-search/floating_table_test.xlsx",
    "/Users/saiteja/Documents/Dev/EDMS/src/smart-search/hidden_sheet_test.xlsx",
    "/Users/saiteja/Documents/Dev/EDMS/src/smart-search/merged_header_test.xlsx",
    "/Users/saiteja/Documents/Dev/EDMS/scripts/stage2_section_context.xlsx",
    # "/Users/saiteja/Downloads/Financial Statements.xlsx",
    # "/Users/saiteja/Downloads/testamazon.xlsx",
]
# ============================================================

SEP = "=" * 70

for filepath in FILES:
    print(f"\n{SEP}")
    print(f"  FILE: {filepath}")
    print(SEP)

    structure = parser.extract_structure(filepath)
    print(f"  Title: {structure.title}")
    print(f"  Pages (sheets): {len(structure.pages)}")

    for page in structure.pages:
        print(f"\n  --- Page {page.page_number} ({len(page.sections)} sections) ---")
        for sec in page.sections:
            rtype = sec.section_type
            level = sec.level
            title = sec.title[:35]
            content = sec.content[:150].replace("\n", " ")
            print(f"    [{rtype:7s}] [L{level}] [{title:35s}] {content}")

print(f"\n{SEP}")
print("  DONE")
print(SEP)
