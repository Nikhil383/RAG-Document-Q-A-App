"""End-to-end multimodal RAG pipeline: ingest documents, then query them."""

from __future__ import annotations

from .config import CONFIG
from .document_processor import DocumentProcessor, Document
from .embeddings import ClipEmbedder
from .vector_store import MultimodalVectorStore
from .retriever import MultimodalRetriever
from .generator import AnswerGenerator


class MultimodalRAG:
    def __init__(self, config=CONFIG, lazy_generator: bool = True):
        self.config = config
        self.processor = DocumentProcessor(config)
        self.embedder = ClipEmbedder(config)
        self.store = MultimodalVectorStore(config)
        self.retriever = MultimodalRetriever(self.embedder, self.store, config)
        # Generator needs an API key; don't fail construction if it's absent
        # unless the caller explicitly wants it eagerly.
        self._generator = None if lazy_generator else AnswerGenerator(config)

    @property
    def generator(self) -> AnswerGenerator:
        if self._generator is None:
            self._generator = AnswerGenerator(self.config)
        return self._generator

    # ---------------- ingestion ----------------

    def ingest_pdf(self, path: str) -> int:
        docs = self.processor.process_pdf(path)
        self._embed_and_store(docs)
        return len(docs)

    def ingest_image_folder(self, folder: str) -> int:
        docs = self.processor.process_image_folder(folder)
        self._embed_and_store(docs)
        return len(docs)

    def ingest_image_file(self, path: str) -> int:
        docs = self.processor.process_image_file(path)
        self._embed_and_store(docs)
        return len(docs)

    def ingest_text_file(self, path: str) -> int:
        docs = self.processor.process_text_file(path)
        self._embed_and_store(docs)
        return len(docs)

    def _embed_and_store(self, docs: list[Document]) -> None:
        text_docs = [d for d in docs if d.type == "text"]
        image_docs = [d for d in docs if d.type == "image"]

        if text_docs:
            embeddings = self.embedder.embed_text([d.content for d in text_docs])
            self.store.add(text_docs, embeddings)

        if image_docs:
            embeddings = self.embedder.embed_images([d.content for d in image_docs])
            self.store.add(image_docs, embeddings)

    # ---------------- querying ----------------

    def query(self, question: str, top_k: int | None = None) -> dict:
        retrieved = self.retriever.retrieve(question, top_k=top_k)
        answer = self.generator.generate(question, retrieved)
        return {
            "answer": answer,
            "sources": {
                "text": [
                    {"source": h["metadata"].get("source"), "page": h["metadata"].get("page")}
                    for h in retrieved["text"]
                ],
                "images": [
                    {"source": h["metadata"].get("source"), "page": h["metadata"].get("page"), "path": h["content"]}
                    for h in retrieved["images"]
                ],
            },
        }

    def stats(self) -> dict:
        return {"total_indexed": self.store.count()}
