import fitz  # PyMuPDF
import os

# 1. Extract text from PDF
def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

# 2. Extract content from PDF in markdown
def extract_markdown(pdf_path):
    doc = fitz.open(pdf_path)
    markdown = ""
    for page_num, page in enumerate(doc, 1):
        markdown += f"# Page {page_num}\n"
        markdown += page.get_text("markdown")
        markdown += "\n"
    doc.close()
    return markdown

# 3. OCR on an image (requires pytesseract and PIL)
def ocr_image(image_path):
    try:
        from PIL import Image
        import pytesseract
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text
    except ImportError:
        print("Please install pytesseract and pillow for OCR functionality.")
        return None

# Example usage (uncomment and set paths to use)
if __name__ == "__main__":
    # pdf_file = "input.pdf"
    # print(extract_text(pdf_file))
    # print(extract_markdown(pdf_file))
    # image_file = "image.png"
    # print(ocr_image(image_file))
    pass
