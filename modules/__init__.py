"""RAG Document Processing Modules.

This package contains the core components for the RAG application:
- PDF processing and text extraction
- Text chunking and embedding generation
- Vector retrieval using ChromaDB
- Agentic LLM for question answering
"""

from modules.ingestion.pdf_reader import read_pdf, read_pdf_with_metadata, extract_pdf_info
from modules.ingestion.chunker import chunk_text, chunk_text_with_metadata, chunk_text_semantic
from modules.retrieval.embedder import get_embedding
from modules.retrieval.retriever import Retriever
from modules.retrieval.vector_store import ChromaVectorStore, get_vector_store
from modules.agent.agent import Agent

__all__ = [
    "read_pdf",
    "read_pdf_with_metadata",
    "extract_pdf_info",
    "chunk_text",
    "chunk_text_with_metadata",
    "chunk_text_semantic",
    "get_embedding",
    "Retriever",
    "ChromaVectorStore",
    "get_vector_store",
    "Agent",
]
