import faiss
import numpy as np
from modules.embedder import get_embedding

class Retriever:
    def __init__(self, chunks, embeddings):
        self.chunks = chunks
        self.embeddings = np.array(embeddings).astype("float32")
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def search(self, query_embedding, top_k=3):
        D, I = self.index.search(np.array([query_embedding]).astype("float32"), top_k)
        return I[0]

    def query(self, query_text: str, top_k: int = 3) -> str:
        """Retrieves relevant context for a given query text."""
        query_embedding = get_embedding(query_text)
        indices = self.search(query_embedding, top_k=top_k)
        results = [self.chunks[i] for i in indices]
        return "\n\n".join(results)
