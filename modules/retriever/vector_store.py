from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

class VectorStore:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        self.db = None

    def build(self, docs):
        self.db = FAISS.from_documents(docs, self.embeddings)

    def search(self, query, k=5):
        if not self.db:
            return []
        return self.db.similarity_search(query, k=k)