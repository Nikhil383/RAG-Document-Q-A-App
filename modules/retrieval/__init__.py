"""Retrieval module for embedding generation and vector search."""

from modules.retrieval.embedder import get_embedding
from modules.retrieval.retriever import Retriever

__all__ = ["get_embedding", "Retriever"]