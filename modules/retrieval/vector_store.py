import os
import uuid
from typing import List, Dict, Any, Optional
from datetime import datetime

import chromadb
from chromadb.config import Settings
from dotenv import load_dotenv

from modules.retrieval.embedder import get_embedding, get_embeddings

load_dotenv()


class ChromaVectorStore:
    """Persistent vector store using ChromaDB for document embeddings."""

    def __init__(self, persist_directory: str = "./chroma_db"):
        """Initialize ChromaDB client with persistence.

        Args:
            persist_directory: Directory to store ChromaDB data
        """
        self.persist_directory = persist_directory

        # Initialize ChromaDB client with persistence
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            )
        )

    def create_collection(self, session_id: str) -> chromadb.Collection:
        """Create or get a collection for a session.

        Args:
            session_id: Unique session identifier

        Returns:
            ChromaDB collection
        """
        collection_name = f"session_{session_id}"

        # Try to get existing collection, or create new one
        try:
            collection = self.client.get_collection(name=collection_name)
        except Exception:
            collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}  # Use cosine similarity
            )

        return collection

    def add_documents(
        self,
        session_id: str,
        chunks: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        document_id: Optional[str] = None
    ) -> List[str]:
        """Add document chunks to the vector store.

        Args:
            session_id: Session identifier
            chunks: List of text chunks
            metadata: Optional metadata for each chunk (page numbers, source, etc.)
            document_id: Optional document identifier for grouping

        Returns:
            List of chunk IDs
        """
        collection = self.create_collection(session_id)

        # Generate embeddings for all chunks in a single batch call
        embeddings = get_embeddings(chunks)

        # Generate unique IDs for each chunk
        chunk_ids = [str(uuid.uuid4()) for _ in chunks]

        # Prepare metadata
        if metadata is None:
            metadata = [{} for _ in chunks]

        # Add document_id to each metadata entry if provided
        for i, meta in enumerate(metadata):
            meta["chunk_index"] = i
            meta["timestamp"] = datetime.now().isoformat()
            if document_id:
                meta["document_id"] = document_id
                meta["document_name"] = document_id  # For display purposes

        # Add to ChromaDB
        collection.add(
            ids=chunk_ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadata
        )

        return chunk_ids

    def query(
        self,
        session_id: str,
        query_text: str,
        top_k: int = 3,
        filter_dict: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """Query the vector store for relevant chunks.

        Args:
            session_id: Session identifier
            query_text: Query text
            top_k: Number of results to return
            filter_dict: Optional metadata filter

        Returns:
            List of result dictionaries with text, metadata, and distance
        """
        collection = self.create_collection(session_id)

        # Get query embedding
        query_embedding = get_embedding(query_text)

        # Query ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=filter_dict,
            include=["documents", "metadatas", "distances"]
        )

        # Format results
        formatted_results = []
        if results["ids"] and results["ids"][0]:
            for i, doc_id in enumerate(results["ids"][0]):
                formatted_results.append({
                    "id": doc_id,
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0
                })

        return formatted_results

    def get_collection_stats(self, session_id: str) -> Dict[str, Any]:
        """Get statistics about a session's collection.

        Args:
            session_id: Session identifier

        Returns:
            Dictionary with collection statistics
        """
        try:
            collection = self.client.get_collection(name=f"session_{session_id}")
            count = collection.count()
            return {"exists": True, "document_count": count}
        except Exception:
            return {"exists": False, "document_count": 0}

    def delete_collection(self, session_id: str) -> bool:
        """Delete a session's collection.

        Args:
            session_id: Session identifier

        Returns:
            True if deleted, False otherwise
        """
        try:
            self.client.delete_collection(name=f"session_{session_id}")
            return True
        except Exception:
            return False

    def delete_document(self, session_id: str, document_id: str) -> bool:
        """Delete all chunks belonging to a specific document.

        Args:
            session_id: Session identifier
            document_id: Document identifier

        Returns:
            True if deleted, False otherwise
        """
        try:
            collection = self.client.get_collection(name=f"session_{session_id}")
            collection.delete(where={"document_id": document_id})
            return True
        except Exception:
            return False

    def get_documents_in_session(self, session_id: str) -> List[Dict[str, Any]]:
        """Get list of unique documents in a session.

        Args:
            session_id: Session identifier

        Returns:
            List of document info dictionaries
        """
        try:
            collection = self.client.get_collection(name=f"session_{session_id}")
            results = collection.get(include=["metadatas"])

            # Extract unique document IDs
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


# Global vector store instance
_vector_store: Optional[ChromaVectorStore] = None


def get_vector_store(persist_directory: str = "./chroma_db") -> ChromaVectorStore:
    """Get or create the global vector store instance.

    Args:
        persist_directory: Directory for ChromaDB persistence

    Returns:
        ChromaVectorStore instance
    """
    global _vector_store
    if _vector_store is None:
        _vector_store = ChromaVectorStore(persist_directory)
    return _vector_store
