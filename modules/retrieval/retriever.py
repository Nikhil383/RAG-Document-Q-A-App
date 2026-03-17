from typing import List, Dict, Any, Optional

from modules.retrieval.vector_store import ChromaVectorStore, get_vector_store


class Retriever:
    """Document retriever using ChromaDB for vector search."""

    def __init__(
        self,
        session_id: str,
        vector_store: Optional[ChromaVectorStore] = None
    ):
        """Initialize retriever for a session.

        Args:
            session_id: Unique session identifier
            vector_store: Optional vector store instance (uses global if not provided)
        """
        self.session_id = session_id
        self.vector_store = vector_store or get_vector_store()

    def add_documents(
        self,
        chunks: List[str],
        metadata: Optional[List[Dict[str, Any]]] = None,
        document_id: Optional[str] = None
    ) -> List[str]:
        """Add document chunks to the retriever.

        Args:
            chunks: List of text chunks
            metadata: Optional metadata for each chunk
            document_id: Optional document identifier

        Returns:
            List of chunk IDs
        """
        return self.vector_store.add_documents(
            session_id=self.session_id,
            chunks=chunks,
            metadata=metadata,
            document_id=document_id
        )

    def query(
        self,
        query_text: str,
        top_k: int = 3,
        document_id: Optional[str] = None
    ) -> str:
        """Retrieve relevant context for a given query text.

        Args:
            query_text: The query text
            top_k: Number of chunks to retrieve
            document_id: Optional specific document to search

        Returns:
            Concatenated relevant chunks as a string
        """
        # Build filter if document_id specified
        filter_dict = None
        if document_id:
            filter_dict = {"document_id": document_id}

        results = self.vector_store.query(
            session_id=self.session_id,
            query_text=query_text,
            top_k=top_k,
            filter_dict=filter_dict
        )

        # Format results with citation info
        formatted_chunks = []
        for result in results:
            text = result["text"]
            meta = result.get("metadata", {})

            # Add citation info if available
            citation_parts = []
            if meta.get("page_number"):
                citation_parts.append(f"Page {meta['page_number']}")
            if meta.get("document_name"):
                citation_parts.append(f"from {meta['document_name']}")

            if citation_parts:
                text = f"[{', '.join(citation_parts)}]\n{text}"

            formatted_chunks.append(text)

        return "\n\n---\n\n".join(formatted_chunks)

    def query_with_sources(
        self,
        query_text: str,
        top_k: int = 3,
        document_id: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Query and return results with full source information.

        Args:
            query_text: The query text
            top_k: Number of chunks to retrieve
            document_id: Optional specific document to search

        Returns:
            List of result dictionaries with text, metadata, and distance
        """
        filter_dict = None
        if document_id:
            filter_dict = {"document_id": document_id}

        return self.vector_store.query(
            session_id=self.session_id,
            query_text=query_text,
            top_k=top_k,
            filter_dict=filter_dict
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get retriever statistics.

        Returns:
            Dictionary with statistics
        """
        return self.vector_store.get_collection_stats(self.session_id)

    def get_documents(self) -> List[Dict[str, Any]]:
        """Get list of documents in this session.

        Returns:
            List of document info dictionaries
        """
        return self.vector_store.get_documents_in_session(self.session_id)

    def delete_document(self, document_id: str) -> bool:
        """Delete a specific document from the session.

        Args:
            document_id: Document identifier

        Returns:
            True if deleted, False otherwise
        """
        return self.vector_store.delete_document(self.session_id, document_id)
