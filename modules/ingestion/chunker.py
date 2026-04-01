import hashlib
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_docs(docs, chunk_size=500, chunk_overlap=50):
    """Split documents into chunks with metadata"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = []
    for i, doc in enumerate(docs):
        # Generate a unique document ID
        doc_id = hashlib.md5(doc.page_content[:100].encode()).hexdigest()[:8]
        
        # Split the document
        doc_chunks = splitter.split_documents([doc])
        
        for j, chunk in enumerate(doc_chunks):
            # Add metadata
            chunk.metadata["doc_id"] = doc_id
            chunk.metadata["chunk_idx"] = j
            chunk.metadata["page"] = doc.metadata.get("page", 0)
            chunks.append(chunk)
    
    return chunks
