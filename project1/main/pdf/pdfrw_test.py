from pdfrw import PdfReader, PdfWriter
import os

def read_pdf(input_path):
    reader = PdfReader(input_path)
    return reader

def write_pdf(reader, output_path):
    writer = PdfWriter()
    writer.addpages(reader.pages)
    writer.write(output_path)

# Example usage (uncomment and set paths to use)
if __name__ == "__main__":
    # input_pdf = "input.pdf"
    # output_pdf = "output.pdf"
    # reader = read_pdf(input_pdf)
    # write_pdf(reader, output_pdf)
    pass
