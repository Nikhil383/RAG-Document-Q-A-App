from typing import List, Dict, Any, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(
    text: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200
) -> List[str]:
    """Split text into semantic chunks using LangChain's RecursiveCharacterTextSplitter.

    Args:
        text: Input text to chunk
        chunk_size: Maximum size of each chunk (default 1000)
        chunk_overlap: Overlap between chunks to preserve context (default 200)

    Returns:
        List of text chunks
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""]  # Split by paragraphs, sentences, words
    )
    return splitter.split_text(text)


def chunk_text_with_metadata(
    pages: List[Dict[str, Any]],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    document_id: Optional[str] = None
) -> List[Dict[str, Any]]:
    """Split text into chunks while preserving page metadata.

    Args:
        pages: List of page dictionaries with 'text' and 'metadata' keys
        chunk_size: Maximum size of each chunk
        chunk_overlap: Overlap between chunks
        document_id: Optional document ID to add to metadata

    Returns:
        List of chunk dictionaries with 'text' and 'metadata'
    """
    chunks_with_metadata = []

    for page in pages:
        page_text = page["text"]
        page_metadata = page.get("metadata", {})

        # Skip empty pages
        if not page_text.strip():
            continue

        # Create a splitter for this page
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        # Split the page text
        text_chunks = splitter.split_text(page_text)

        # Create chunk entries with metadata
        for i, chunk_text in enumerate(text_chunks):
            chunk_metadata = {
                **page_metadata,
                "chunk_index": i,
                "chunk_in_page": i + 1,
            }

            if document_id:
                chunk_metadata["document_id"] = document_id

            chunks_with_metadata.append({
                "text": chunk_text,
                "metadata": chunk_metadata
            })

    return chunks_with_metadata


def chunk_text_semantic(
    text: str,
    target_chunk_size: int = 1000,
    min_chunk_size: int = 200
) -> List[str]:
    """Split text into semantic chunks using paragraph boundaries.

    This is a simpler semantic chunking approach that respects paragraph
    boundaries while trying to hit the target chunk size.

    Args:
        text: Input text to chunk
        target_chunk_size: Target size for each chunk
        min_chunk_size: Minimum size for a chunk

    Returns:
        List of text chunks
    """
    paragraphs = text.split("\n\n")
    chunks = []
    current_chunk = []
    current_size = 0

    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue

        paragraph_size = len(paragraph)

        # If adding this paragraph would exceed target, start new chunk
        if current_size + paragraph_size > target_chunk_size and current_chunk:
            chunks.append("\n\n".join(current_chunk))
            current_chunk = [paragraph]
            current_size = paragraph_size
        else:
            current_chunk.append(paragraph)
            current_size += paragraph_size

    # Don't forget the last chunk
    if current_chunk:
        # If last chunk is too small and we have previous chunks, merge with previous
        if current_size < min_chunk_size and len(chunks) > 0:
            chunks[-1] = chunks[-1] + "\n\n" + "\n\n".join(current_chunk)
        else:
            chunks.append("\n\n".join(current_chunk))

    return chunks
