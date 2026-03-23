import os
from typing import List
from langchain_community.document_loaders import PyPDFLoader, CSVLoader
from langchain_core.documents import Document

class DocumentLoader:
    @staticmethod
    def load_file(file_path: str, document_id: str = None) -> List[Document]:
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        if ext == '.pdf':
            loader = PyPDFLoader(file_path)
        elif ext == '.csv':
            loader = CSVLoader(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
            
        docs = loader.load()
        if document_id:
            for doc in docs:
                doc.metadata['document_id'] = document_id
        return docs
