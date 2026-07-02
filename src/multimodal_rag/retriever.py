"""Retrieves the most relevant text chunks and images for a query."""

from __future__ import annotations

from .config import CONFIG
from .embeddings import ClipEmbedder
from .vector_store import MultimodalVectorStore


class MultimodalRetriever:
    def __init__(self, embedder: ClipEmbedder, store: MultimodalVectorStore, config=CONFIG):
        self.embedder = embedder
        self.store = store
        self.config = config

    def retrieve(self, query: str, top_k: int | None = None) -> dict:
        """Returns the top-k text chunks and top-k images ranked separately,
        since mixing modalities into one ranked list under CLIP similarity
        tends to favor one modality unfairly (their score distributions
        differ). Ranking within each modality is reliable; ranking across
        modalities is not.
        """
        top_k = top_k or self.config.top_k
        query_embedding = self.embedder.embed_query(query)

        text_hits = self.store.query(query_embedding, top_k=top_k, type_filter="text")
        image_hits = self.store.query(query_embedding, top_k=max(1, top_k // 2), type_filter="image")

        return {"text": text_hits, "images": image_hits}
