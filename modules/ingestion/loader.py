from langchain_community.document_loaders import PyPDFLoader, CSVLoader, TextLoader
import os

def load_pdf(path):
    """Load PDF document"""
    loader = PyPDFLoader(path)
    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = path
        doc.metadata["type"] = "pdf"
    return docs

def load_csv(path):
    """Load CSV document"""
    loader = CSVLoader(path)
    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = path
        doc.metadata["type"] = "csv"
    return docs

def load_txt(path):
    """Load text document"""
    loader = TextLoader(path)
    docs = loader.load()
    for doc in docs:
        doc.metadata["source"] = path
        doc.metadata["type"] = "txt"
    return docs

def load_document(path):
    """Auto-detect file type and load"""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return load_pdf(path)
    elif ext == ".csv":
        return load_csv(path)
    elif ext in [".txt", ".md"]:
        return load_txt(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
