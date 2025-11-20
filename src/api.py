# src/api.py
import os
import json
import uuid
import shutil
import numpy as np
import pdfplumber
import faiss
from google import genai
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from retriever import Retriever
from generator import generate_answer
import docx2txt

client = genai.Client()

DATA_DIR = "data"
DOCS_JSONL = os.path.join(DATA_DIR, "docs.jsonl")
META_JSONL = os.path.join(DATA_DIR, "meta.jsonl")
INDEX_PATH = os.path.join(DATA_DIR, "faiss.index")

app = FastAPI()
retriever = None

# ------------- UTIL FUNCS ---------------- #

def chunk(text, size=800, overlap=100):
    words = text.split()
    out = []
    i = 0
    while i < len(words):
        out.append(" ".join(words[i:i+size]))
        i += size - overlap
    return out

def embed_texts(texts, batch_size=32):
    vectors = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        res = client.models.embed_content(model="gemini-embedding-001", contents=batch)
        for emb in res.embeddings:
            vectors.append(np.array(emb.vector, dtype="float32"))
    return vectors

def save_index(index):
    faiss.write_index(index, INDEX_PATH)

def load_index():
    if os.path.exists(INDEX_PATH):
        return faiss.read_index(INDEX_PATH)
    return None

def extract_pdf(path):
    pages = []
    with pdfplumber.open(path) as pdf:
        for i, p in enumerate(pdf.pages):
            txt = p.extract_text() or ""
            pages.append((i + 1, txt))
    return pages

def extract_docx(path):
    try:
        return docx2txt.process(path) or ""
    except Exception:
        return ""

# ------------- ENDPOINTS ---------------- #

class Query(BaseModel):
    question: str

@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    """
    Upload a PDF, TXT, or DOCX file.
    Saves file -> extracts text -> chunks -> appends docs -> embeds -> updates FAISS.
    """
    global retriever
    os.makedirs(os.path.join(DATA_DIR, "input_docs"), exist_ok=True)

    orig_filename = file.filename
    ext = orig_filename.lower().split(".")[-1]
    save_path = os.path.join(DATA_DIR, "input_docs", f"{uuid.uuid4()}-{orig_filename}")

    # save uploaded file
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Extract text depending on extension
    text = ""
    if ext == "pdf":
        pages = extract_pdf(save_path)
        # join pages with newlines (we chunk per combined text here)
        text = "\n".join([p_text for _, p_text in pages if p_text])
    elif ext in ("txt", "md"):
        try:
            with open(save_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception:
            text = ""
    elif ext == "docx":
        text = extract_docx(save_path)
    elif ext == "doc":
        # .doc binary fallback — try to read as text, but not guaranteed
        try:
            with open(save_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        except Exception:
            text = ""
    else:
        return {"status": "error", "message": f"Unsupported file type: .{ext}"}

    # If no useful text extracted
    if not text or len(text.split()) < 20:
        return {"status": "ok", "message": "No extractable text found or file too small.", "added_chunks": 0}

    # Chunk text
    chunks = chunk(text)
    new_docs = []
    new_meta = []
    for idx, c in enumerate(chunks):
        doc = {
            "id": str(uuid.uuid4()),
            "text": c,
            "metadata": {"source": orig_filename, "path": save_path, "chunk": idx}
        }
        new_docs.append(doc)
        new_meta.append(doc["metadata"])

    # Append to docs.jsonl and meta.jsonl
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(DOCS_JSONL, "a", encoding="utf-8") as f:
        for d in new_docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    with open(META_JSONL, "a", encoding="utf-8") as f:
        for m in new_meta:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")

    # Embed and update FAISS
    vectors = embed_texts([d["text"] for d in new_docs])
    index = load_index()
    if index is None:
        dim = vectors[0].shape[0]
        index = faiss.IndexFlatL2(dim)
        index.add(np.vstack(vectors))
    else:
        index.add(np.vstack(vectors))
    save_index(index)

    retriever = None  # force reload on next query

    return {"status": "ok", "message": f"Uploaded & indexed {orig_filename}", "added_chunks": len(new_docs)}


@app.post("/answer")
def answer(q: Query):
    global retriever
    if retriever is None:
        retriever = Retriever()
    chunks = retriever.retrieve(q.question, k=4)
    ans = generate_answer(q.question, chunks)
    return {"answer": ans, "sources": [c.get("metadata", {}) for c in chunks]}


@app.get("/")
def health():
    return {"status": "running"}
