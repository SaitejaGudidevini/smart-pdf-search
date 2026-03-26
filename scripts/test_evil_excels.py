"""Run the Excel parser against the 3 evil test files and show output."""

import sys
sys.path.insert(0, "/Users/saiteja/Documents/Dev/EDMS/src/smart-search")

from excel_parser import ExcelParser
from excel_enricher import ExcelEnricher

parser = ExcelParser()
enricher = ExcelEnricher()

TEST_DIR = "/Users/saiteja/Documents/Dev/EDMS/scripts/test_excels"

SEPARATOR = "=" * 70


def test_file(label: str, filepath: str):
    print(f"\n{SEPARATOR}")
    print(f"  TEST: {label}")
    print(f"  FILE: {filepath}")
    print(SEPARATOR)

    # Stage 1: Parse
    dataframes, formulas = parser.parse(filepath)

    for sheet_name, df in dataframes.items():
        print(f"\n--- Sheet: '{sheet_name}' ---")
        print(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
        print(f"Columns: {list(df.columns)}")
        print(f"Dtypes:\n{df.dtypes.to_string()}")
        print(f"\nData:")
        print(df.to_string())

        sheet_formulas = formulas.get(sheet_name, [])
        if sheet_formulas:
            print(f"\nFormulas: {sheet_formulas}")

    # Stage 1b: Schema description
    print(f"\n--- Auto Schema Description ---")
    print(enricher.generate_all_schemas(dataframes))

    # Stage 1c: Semantic rows (first 3)
    print(f"\n--- Semantic Rows (first 3 per sheet) ---")
    for sheet_name, df in dataframes.items():
        rows = enricher.generate_semantic_rows(df, sheet_name, formulas.get(sheet_name, []))
        for row in rows[:3]:
            print(f"  {row['text']}")
        if len(rows) > 3:
            print(f"  ... ({len(rows) - 3} more rows)")

    # Stage 1d: DocumentStructure
    structure = parser.extract_structure(filepath)
    print(f"\n--- DocumentStructure ---")
    print(f"Title: {structure.title}")
    print(f"Doc type: {structure.doc_type}")
    print(f"Pages (sheets): {len(structure.pages)}")
    for page in structure.pages:
        print(f"  Page {page.page_number}: {len(page.sections)} sections")
        for sec in page.sections[:3]:
            preview = sec.content[:100].replace("\n", " ")
            print(f"    [{sec.level}] {sec.title}: {preview}...")


if __name__ == "__main__":
    print("EVIL SPREADSHEET PARSER TEST SUITE")
    print("Testing: ExcelParser + ExcelEnricher\n")

    test_file(
        "1. FLOATING TABLE — data starts at C5, not A1",
        f"{TEST_DIR}/evil_1_floating_table.xlsx",
    )

    test_file(
        "2. MERGED HEADERS — 'Contact Info' spans Phone + Email",
        f"{TEST_DIR}/evil_2_merged_headers.xlsx",
    )

    test_file(
        "3. HIDDEN SHEET — 'Internal Costs' is hidden with sensitive data",
        f"{TEST_DIR}/evil_3_hidden_sheet.xlsx",
    )

    print(f"\n{SEPARATOR}")
    print("  ALL TESTS COMPLETE")
    print(SEPARATOR)
