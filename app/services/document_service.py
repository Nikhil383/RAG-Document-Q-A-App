import uuid
from typing import Optional
from app.ingestion.loaders import DocumentLoader
from app.retrieval.hybrid_search import hybrid_retriever

class DocumentService:
    @staticmethod
    def process_file(file_path: str) -> dict:
        doc_id = str(uuid.uuid4())
        
        # Load Raw Docs (includes parent document creation contextually)
        docs = DocumentLoader.load_file(file_path, document_id=doc_id)
        
        # The hybrid retriever will handle parent splitting and child adding underneath
        hybrid_retriever.add_documents(docs)
        
        return {
            "document_id": doc_id,
            "chunks": len(docs),
            "message": "File processed and indexed successfully."
        }
