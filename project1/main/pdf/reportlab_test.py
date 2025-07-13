from reportlab.pdfgen import canvas
import os

def write_text_to_pdf(output_path, text):
    c = canvas.Canvas(output_path)
    c.drawString(100, 750, text)
    c.save()

# Example usage (uncomment and set paths to use)
if __name__ == "__main__":
    # output_pdf = "output.pdf"
    # write_text_to_pdf(output_pdf, "Hello, ReportLab PDF!")
    pass
