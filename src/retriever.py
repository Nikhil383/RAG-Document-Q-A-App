import json, numpy as np
import faiss
from google import genai

client = genai.Client()

class Retriever:
    def __init__(self):
        self.index = faiss.read_index("data/faiss.index")
        self.docs = [json.loads(l) for l in open("data/docs.jsonl")]

    def embed(self, text):
        r = client.models.embed_content(
            model="gemini-embedding-001",
            contents=[text]
        )
        return np.array(r.embeddings[0].vector, dtype="float32")

    def retrieve(self, query, k=4):
        qv = self.embed(query).reshape(1, -1)
        D, I = self.index.search(qv, k)
        return [self.docs[i] for i in I[0]]

if __name__ == "__main__":
    r = Retriever()
    print(r.retrieve("How do I reset password?"))
