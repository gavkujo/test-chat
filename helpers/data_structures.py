from io import BytesIO
import PyPDF2

def str_parser(string, delimiter, ignore_spaces=True):
    output = list()
    if ignore_spaces:
        string = string.replace(" ", "")
    else:
        string = string
    return string.split(delimiter)

def merge_pdfs_to_bytes(merged_pdf: PyPDF2.PdfMerger) -> bytes:
    output_buffer = BytesIO()
    merged_pdf.write(output_buffer)
    return output_buffer.getvalue()