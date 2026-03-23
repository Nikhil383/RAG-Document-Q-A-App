from langchain_text_splitters import RecursiveCharacterTextSplitter
from app.core.config import settings

class Chunker:
    @staticmethod
    def get_parent_splitter():
        return RecursiveCharacterTextSplitter(
            chunk_size=settings.PARENT_CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP
        )

    @staticmethod
    def get_child_splitter():
        return RecursiveCharacterTextSplitter(
            chunk_size=settings.CHILD_CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP // 2
        )
