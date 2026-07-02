"""Vector storage and similarity search on top of ChromaDB."""

from __future__ import annotations

import numpy as np
import chromadb

from .config import CONFIG
from .document_processor import Document


class MultimodalVectorStore:
    def __init__(self, config=CONFIG):
        self.client = chromadb.PersistentClient(path=config.persist_dir)
        self.collection = self.client.get_or_create_collection(
            name=config.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(self, docs: list[Document], embeddings: np.ndarray) -> None:
        if not docs:
            return
        self.collection.add(
            ids=[d.id for d in docs],
            embeddings=embeddings.tolist(),
            documents=[d.content for d in docs],
            metadatas=[{**d.metadata, "type": d.type} for d in docs],
        )

    def query(self, embedding: np.ndarray, top_k: int, type_filter: str | None = None):
        where = {"type": type_filter} if type_filter else None
        results = self.collection.query(
            query_embeddings=[embedding.tolist()],
            n_results=top_k,
            where=where,
        )
        hits = []
        for i in range(len(results["ids"][0])):
            hits.append(
                {
                    "id": results["ids"][0][i],
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "distance": results["distances"][0][i],
                }
            )
        return hits

    def count(self) -> int:
        return self.collection.count()
