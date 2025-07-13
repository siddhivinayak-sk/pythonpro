from pypdf import PdfReader, PdfWriter
import os

# 1. Split PDF file
def split_pdf(input_path, output_dir):
    reader = PdfReader(input_path)
    for i, page in enumerate(reader.pages):
        writer = PdfWriter()
        writer.add_page(page)
        out_path = os.path.join(output_dir, f"page_{i+1}.pdf")
        with open(out_path, "wb") as out_f:
            writer.write(out_f)
    print(f"Split PDF into {len(reader.pages)} pages.")

# 2. Merge PDF files
def merge_pdfs(pdf_paths, output_path):
    writer = PdfWriter()
    for path in pdf_paths:
        reader = PdfReader(path)
        for page in reader.pages:
            writer.add_page(page)
    with open(output_path, "wb") as out_f:
        writer.write(out_f)
    print(f"Merged {len(pdf_paths)} PDFs into {output_path}.")

# 3. Strip text content from PDF file
def extract_text(input_path):
    reader = PdfReader(input_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() or ""
    return text

# 4. Extract images from PDF file
def extract_images(input_path, output_dir):
    reader = PdfReader(input_path)
    img_count = 0
    for i, page in enumerate(reader.pages):
        if '/XObject' in page['/Resources']:
            xObject = page['/Resources']['/XObject']
            for obj in xObject:
                xobj = xObject[obj]
                if xobj['/Subtype'] == '/Image':
                    img_count += 1
                    data = xobj.get_data()
                    ext = 'jpg' if xobj['/Filter'] == '/DCTDecode' else 'png'
                    out_path = os.path.join(output_dir, f"page_{i+1}_img_{img_count}.{ext}")
                    with open(out_path, "wb") as img_f:
                        img_f.write(data)
    print(f"Extracted {img_count} images.")

# Example usage (uncomment and set paths to use)
if __name__ == "__main__":
    # split_pdf('input.pdf', 'output_dir')
    # merge_pdfs(['file1.pdf', 'file2.pdf'], 'merged.pdf')
    # print(extract_text('input.pdf'))
    # extract_images('input.pdf', 'output_dir')
    pass
