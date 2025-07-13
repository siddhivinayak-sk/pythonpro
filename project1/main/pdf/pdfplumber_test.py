import pdfplumber
import os

# 1. Extract text from PDF
def extract_text(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

# 2. Extract tables from PDF
def extract_tables(pdf_path):
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_tables = page.extract_tables()
            tables.extend(page_tables)
    return tables

# Example usage (uncomment and set paths to use)
if __name__ == "__main__":
    # pdf_file = "input.pdf"
    # print(extract_text(pdf_file))
    # tables = extract_tables(pdf_file)
    # for i, table in enumerate(tables):
    #     print(f"Table {i+1}:")
    #     for row in table:
    #         print(row)
    pass
