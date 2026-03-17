from typing import List, Dict, Any
from pypdf import PdfReader


def read_pdf(uploaded_file) -> str:
    """Extract raw text from a PDF file.

    Args:
        uploaded_file: Path to PDF file or file-like object

    Returns:
        Concatenated text from all pages
    """
    reader = PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text


def read_pdf_with_metadata(uploaded_file, document_name: str = None) -> List[Dict[str, Any]]:
    """Extract text from PDF with page-level metadata for citations.

    Args:
        uploaded_file: Path to PDF file or file-like object
        document_name: Name of the document for metadata

    Returns:
        List of dictionaries with 'text' and 'metadata' keys
    """
    reader = PdfReader(uploaded_file)
    pages = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text()
        if text.strip():  # Only include non-empty pages
            pages.append({
                "text": text,
                "metadata": {
                    "page_number": page_num,
                    "document_name": document_name or "Unknown",
                    "total_pages": len(reader.pages)
                }
            })

    return pages


def extract_pdf_info(uploaded_file) -> Dict[str, Any]:
    """Extract basic information about a PDF.

    Args:
        uploaded_file: Path to PDF file or file-like object

    Returns:
        Dictionary with PDF metadata
    """
    reader = PdfReader(uploaded_file)

    return {
        "num_pages": len(reader.pages),
        "metadata": {
            k: str(v) for k, v in (reader.metadata or {}).items()
        }
    }
