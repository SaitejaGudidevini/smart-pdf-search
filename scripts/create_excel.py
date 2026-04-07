import sys
from markitdown import MarkItDown

def select_and_parse_excel():
    if len(sys.argv) < 2:
        print("Usage: python create_excel.py <path_to_excel_file>")
        return

    file_path = sys.argv[1]

    print(f"Processing file: {file_path}...\n")

    # Initialize Microsoft's MarkItDown
    md = MarkItDown()

    # Convert the Excel file
    try:
        result = md.convert(file_path)
        
        print("======================================================")
        print("        HOW YOUR RAG/LLM SEES THE EXCEL DATA          ")
        print("======================================================\n")
        
        # This text_content is what you will chunk and save to pgvector
        print(result.text_content)
        
    except Exception as e:
        print(f"Error processing file: {e}")

if __name__ == "__main__":
    select_and_parse_excel()