# backend/rag_engine.py
import os
import numpy as np
import faiss
import warnings
from pathlib import Path
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# ✅ Version-safe import for LangChain text splitter
try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
except ModuleNotFoundError:
    from langchain_text_splitters import RecursiveCharacterTextSplitter

from PyPDF2 import PdfReader
from docx import Document

warnings.filterwarnings("ignore")


class RAGPipeline:
    def __init__(self):
        self.chunks = []
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.generator = pipeline("text2text-generation", model="google/flan-t5-base")
        self.index = None
        print("[INFO] RAG Engine initialized (waiting for document upload)")

    def load_document(self, file_path: str):
        """
        Loads and processes a document (PDF, DOCX, TXT) and builds FAISS index.
        """
        ext = Path(file_path).suffix.lower()
        text = ""

        if ext == ".pdf":
            reader = PdfReader(file_path)
            for page in reader.pages:
                text += page.extract_text() or ""
        elif ext == ".docx":
            doc = Document(file_path)
            text = "\n".join([p.text for p in doc.paragraphs])
        elif ext == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            raise ValueError("Unsupported file format. Please upload PDF, DOCX, or TXT.")

        if not text.strip():
            raise ValueError("The document appears to be empty or unreadable.")

        print(f"[INFO] Loaded document with {len(text)} characters.")

        # --- Split Text into Chunks ---
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=200,
            chunk_overlap=30,
            length_function=len
        )
        self.chunks = splitter.split_text(text)
        print(f"[INFO] Text split into {len(self.chunks)} chunks.")

        # --- Embed and Index ---
        chunk_embeddings = self.model.encode(self.chunks)
        d = chunk_embeddings.shape[1]
        self.index = faiss.IndexFlatL2(d)
        self.index.add(np.array(chunk_embeddings).astype("float32"))

        print(f"[INFO] FAISS index built with {self.index.ntotal} vectors.")

    def answer(self, query: str) -> str:
        """
        Generates an answer using the currently loaded document.
        """
        if not self.index:
            return "⚠️ No document uploaded yet. Please upload a file first."
        if not query.strip():
            return "Please enter a valid question."

        # --- Retrieve Top Relevant Chunks ---
        query_embedding = self.model.encode([query]).astype("float32")
        distances, indices = self.index.search(query_embedding, k=2)
        retrieved_chunks = [self.chunks[i] for i in indices[0]]
        context = "\n\n".join(retrieved_chunks)

        # --- Build Prompt ---
        prompt = f"""
        Answer the following question using *only* the provided context.
        If the answer is not in the context, say "I don't have that information."

        Context:
        {context}

        Question:
        {query}

        Answer:
        """

        # --- Generate Answer ---
        try:
            response = self.generator(prompt, max_length=150)
            return response[0]["generated_text"].strip()
        except Exception as e:
            return f"Error generating answer: {str(e)}"
