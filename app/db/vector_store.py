from typing import List, Dict, Any
import os
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.core.config import settings

def get_embeddings():
    if not settings.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY is not set.")
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=settings.GEMINI_API_KEY
    )

class VectorStore:
    def __init__(self, persist_directory: str = settings.CHROMA_PERSIST_DIR):
        self.persist_directory = persist_directory
        self.embeddings = get_embeddings()
        self._client = None

    @property
    def client(self):
        import chromadb
        if self._client is None:
            self._client = chromadb.PersistentClient(path=self.persist_directory)
        return self._client

    def get_collection(self, session_id: str):
        collection_name = f"session_{session_id}"
        return Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )

    def add_documents(self, documents: List[Any], session_id: str):
        store = self.get_collection(session_id)
        store.add_documents(documents)
        
    def similarity_search(self, query: str, session_id: str, k: int = 4, filter: dict = None) -> List[Any]:
        store = self.get_collection(session_id)
        return store.similarity_search(query, k=k, filter=filter)
        
    def similarity_search_with_score(self, query: str, session_id: str, k: int = 4, filter: dict = None):
        store = self.get_collection(session_id)
        return store.similarity_search_with_relevance_scores(query, k=k, filter=filter)

    def delete_collection(self, session_id: str):
        collection_name = f"session_{session_id}"
        try:
            self.client.delete_collection(collection_name)
        except Exception:
            pass

    def delete_document(self, session_id: str, document_id: str):
        store = self.get_collection(session_id)
        store.delete(where={"document_id": document_id})

    def get_documents_in_session(self, session_id: str) -> List[Dict[str, Any]]:
        try:
            collection_name = f"session_{session_id}"
            col = self.client.get_collection(collection_name)
            results = col.get(include=["metadatas"])
            
            documents = {}
            if results["metadatas"]:
                for meta in results["metadatas"]:
                    doc_id = meta.get("document_id")
                    if doc_id and doc_id not in documents:
                        documents[doc_id] = {
                            "id": doc_id,
                            "name": meta.get("document_name", doc_id),
                            "uploaded_at": meta.get("timestamp")
                        }
            return list(documents.values())
        except Exception:
            return []

    def get_stats(self, session_id: str) -> Dict[str, Any]:
        try:
            collection_name = f"session_{session_id}"
            col = self.client.get_collection(collection_name)
            return {"document_count": col.count()}
        except Exception:
            return {"document_count": 0}

vector_store = VectorStore()
