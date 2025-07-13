import sys
import os
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

# Function to read markdown and write to PDF
def markdown_to_pdf(md_path, pdf_path):
    if not os.path.isfile(md_path):
        print(f"Markdown file not found: {md_path}")
        return
    with open(md_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter
    y = height - 50
    for line in lines:
        line = line.strip()
        if not line:
            y -= 20
            continue
        # Simple markdown header handling
        if line.startswith('# '):
            c.setFont("Helvetica-Bold", 16)
            c.drawString(50, y, line[2:])
            y -= 30
        elif line.startswith('## '):
            c.setFont("Helvetica-Bold", 14)
            c.drawString(50, y, line[3:])
            y -= 25
        elif line.startswith('### '):
            c.setFont("Helvetica-Bold", 12)
            c.drawString(50, y, line[4:])
            y -= 22
        else:
            c.setFont("Helvetica", 10)
            c.drawString(50, y, line)
            y -= 15
        if y < 50:
            c.showPage()
            y = height - 50
    c.save()
    print(f"PDF created: {pdf_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python markdown_to_pdf.py <input.md> <output.pdf>")
    else:
        markdown_to_pdf(sys.argv[1], sys.argv[2])

