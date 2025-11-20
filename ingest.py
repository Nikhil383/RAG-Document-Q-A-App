# ingest.py
"""
Batch ingestion script for RAG app (Gemini embeddings + FAISS).
Usage:
    python ingest.py
This reads files from ./docs (PDF, .txt, .md), splits into chunks,
creates embeddings via Google Generative AI (Gemini) and saves a FAISS index
to ./faiss_index.
"""
from pathlib import Path
import os
import sys

# LangChain-community + new-style imports
from langchain_community.document_loaders import TextLoader, UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

# Gemini (Google GenAI) integration
from langchain_google_genai import GoogleGenerativeAIEmbeddings

DOCS_DIR = Path("./docs")
STORAGE_DIR = Path("./faiss_index")

def gather_loaders(docs_dir: Path):
    loaders = []
    for p in docs_dir.glob("**/*"):
        if p.is_file():
            suf = p.suffix.lower()
            if suf == ".pdf":
                loaders.append(UnstructuredPDFLoader(str(p)))
            elif suf in (".txt", ".md"):
                loaders.append(TextLoader(str(p), encoding="utf8"))
            else:
                # skip unknown types (or add more loaders)
                print(f"Skipping unsupported file type: {p}", file=sys.stderr)
    return loaders

def ingest_all():
    if not os.getenv("GOOGLE_API_KEY"):
        print("ERROR: Please set GOOGLE_API_KEY environment variable before running.", file=sys.stderr)
        return

    DOCS_DIR.mkdir(exist_ok=True)
    loaders = gather_loaders(DOCS_DIR)

    if not loaders:
        print("No documents found in ./docs. Put PDFs or text files there and try again.", file=sys.stderr)
        return

    # Load documents
    docs = []
    for loader in loaders:
        try:
            loaded = loader.load()
            docs.extend(loaded)
        except Exception as e:
            print(f"Warning: loader failed for {loader}: {e}", file=sys.stderr)

    if not docs:
        print("No content extracted from documents. Exiting.", file=sys.stderr)
        return

    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} document chunks.")

    # Create embeddings (Gemini)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Build FAISS index and persist
    vectordb = FAISS.from_documents(chunks, embedding=embeddings)
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    vectordb.save_local(str(STORAGE_DIR))
    print(f"FAISS index saved to {STORAGE_DIR}")

if __name__ == "__main__":
    ingest_all()
