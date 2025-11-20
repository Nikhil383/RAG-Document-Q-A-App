import json, numpy as np
from tqdm import tqdm
import faiss
from google import genai

client = genai.Client()

def embed_batch(batch):
    res = client.models.embed_content(
        model="gemini-embedding-001",
        contents=batch
    )
    return [np.array(e.vector, dtype="float32") for e in res.embeddings]

def main():
    docs = [json.loads(l) for l in open("data/docs.jsonl")]
    texts = [d["text"] for d in docs]

    vectors = []
    for i in tqdm(range(0, len(texts), 32)):
        vectors.extend(embed_batch(texts[i:i+32]))

    dim = len(vectors[0])
    index = faiss.IndexFlatL2(dim)
    index.add(np.vstack(vectors))
    faiss.write_index(index, "data/faiss.index")

    with open("data/meta.jsonl", "w") as f:
        for d in docs:
            f.write(json.dumps(d["metadata"]) + "\n")

    print("Index built.")

if __name__ == "__main__":
    main()
