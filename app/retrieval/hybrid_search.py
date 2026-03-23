import os
import pickle
from typing import List, Optional
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import ParentDocumentRetriever, EnsembleRetriever
from langchain.storage import InMemoryStore, LocalFileStore
from langchain_core.documents import Document

from app.db.vector_store import get_embeddings, vector_store
from app.ingestion.chunker import Chunker
from app.core.config import settings

class AdvancedRetriever:
    def __init__(self, session_id: str, use_parent_child=True, use_bm25=True):
        self.session_id = session_id
        self.use_parent_child = use_parent_child
        self.use_bm25 = use_bm25
        
        # Persistent store for parent documents
        store_path = os.path.join(settings.CHROMA_PERSIST_DIR, f"parents_{session_id}")
        os.makedirs(os.path.dirname(store_path), exist_ok=True)
        self.store = LocalFileStore(store_path)
        
        self.parent_splitter = Chunker.get_parent_splitter()
        self.child_splitter = Chunker.get_child_splitter()

        self.bm25_retriever: Optional[BM25Retriever] = None
        self.all_documents_for_bm25: List[Document] = []
        
        # Get session-specific vector store
        session_vector_store = vector_store.get_collection(session_id)
        
        if self.use_parent_child:
            self.dense_retriever = ParentDocumentRetriever(
                vectorstore=session_vector_store,
                docstore=self.store,
                child_splitter=self.child_splitter,
                parent_splitter=self.parent_splitter,
            )
        else:
            self.dense_retriever = session_vector_store.as_retriever(search_kwargs={"k": 5})

    def add_documents(self, documents: List[Document]):
        if self.use_parent_child:
            self.dense_retriever.add_documents(documents)
        else:
            self.dense_retriever.vectorstore.add_documents(documents)
            
        if self.use_bm25:
            self.all_documents_for_bm25.extend(documents)
            # Re-initialize BM25 since it's in-memory in this implementation
            # In a larger system, we'd use a dedicated search engine
            self.bm25_retriever = BM25Retriever.from_documents(self.all_documents_for_bm25)
            self.bm25_retriever.k = 5

    def get_retriever(self):
        if self.use_bm25 and self.bm25_retriever:
            return EnsembleRetriever(
                retrievers=[self.bm25_retriever, self.dense_retriever],
                weights=[0.3, 0.7]
            )
        return self.dense_retriever

    def delete_all(self):
        self.all_documents_for_bm25 = []
        self.bm25_retriever = None
        # Clean up local store if needed
        vector_store.delete_collection(self.session_id)

class RetrieverManager:
    def __init__(self):
        self._retrievers: Dict[str, AdvancedRetriever] = {}

    def get_retriever(self, session_id: str) -> AdvancedRetriever:
        if session_id not in self._retrievers:
            self._retrievers[session_id] = AdvancedRetriever(session_id)
        return self._retrievers[session_id]

    def remove_retriever(self, session_id: str):
        if session_id in self._retrievers:
            del self._retrievers[session_id]

hybrid_retriever_manager = RetrieverManager()
