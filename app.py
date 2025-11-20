# app.py
import os
import io
import json
import numpy as np
import streamlit as st
from typing import List, Tuple

# text extraction
from PyPDF2 import PdfReader

# simple text splitter (langchain helper)
try:
    from langchain.text_splitter import CharacterTextSplitter
except Exception:
    # fallback: tiny splitter if langchain import path differs
    class CharacterTextSplitter:
        def __init__(self, chunk_size=1000, chunk_overlap=200):
            self.chunk_size = chunk_size
            self.chunk_overlap = chunk_overlap
        def split_text(self, text):
            chunks = []
            i = 0
            L = len(text)
            while i < L:
                end = min(i + self.chunk_size, L)
                chunks.append(text[i:end])
                i = end - self.chunk_overlap
                if i < 0: i = 0
            return chunks

# FAISS + persistence
import faiss
import numpy as np

# Google GenAI client
# We use the Google Gen AI SDK client interface (create a client object).
try:
    from google import genai
except Exception:
    # If the modern package isn't available, try legacy import (older SDKs)
    try:
        import google.generativeai as genai
    except Exception:
        genai = None

st.set_page_config(page_title="RAG Q&A — Gemini + FAISS", layout="wide")

st.title("RAG Q&A — Google Gemini + FAISS")
st.markdown("""
Upload text / PDF documents, index them (generate embeddings with Gemini),
then ask questions — the app retrieves relevant chunks with FAISS and passes them to Gemini to answer.
""")

################################################################################
# Helper: load text from uploaded files
def extract_text_from_pdf_bytes(b: bytes) -> str:
    try:
        reader = PdfReader(io.BytesIO(b))
        pages = []
        for p in reader.pages:
            try:
                pages.append(p.extract_text() or "")
            except Exception:
                # fallback ignoring extraction errors
                pass
        return "\n".join(pages)
    except Exception as e:
        st.error(f"PDF parse error: {e}")
        return ""

def load_documents(uploaded_files) -> List[Tuple[str, str]]:
    docs = []
    for f in uploaded_files:
        fname = f.name
        bytes_data = f.read()
        if fname.lower().endswith(".pdf"):
            text = extract_text_from_pdf_bytes(bytes_data)
        else:
            # treat as plain text-like
            try:
                text = bytes_data.decode('utf-8')
            except Exception:
                text = bytes_data.decode('latin-1', errors='ignore')
        docs.append((fname, text))
    return docs

################################################################################
# Embedding + FAISS utilities using Google GenAI SDK
def make_genai_client():
    if genai is None:
        raise RuntimeError("google-genai SDK not installed or import failed.")
    # The modern SDK uses genai.Client(); it will read GEMINI_API_KEY or GOOGLE_API_KEY env var
    try:
        client = genai.Client()
    except Exception:
        # older SDKs may require configure
        try:
            genai.configure(api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"))
            client = genai
        except Exception as e:
            raise RuntimeError("Could not initialize Google GenAI client: " + str(e))
    return client

def embed_texts(client, texts: List[str], model: str = "gemini-embedding-001") -> List[np.ndarray]:
    """
    Call Gemini/GenAI embeddings API for a list of texts.
    Returns list of numpy arrays.
    """
    # client.models.embed_content(...) is the modern approach (may vary by SDK version).
    # We handle both possible return shapes.
    try:
        resp = client.models.embed_content(model=model, contents=texts)
        vectors = []
        if isinstance(resp, dict) and "data" in resp:
            for item in resp["data"]:
                vectors.append(np.array(item["embedding"], dtype=np.float32))
        elif isinstance(resp, list):
            for item in resp:
                vectors.append(np.array(item["embedding"], dtype=np.float32))
        elif hasattr(resp, "embeddings") or hasattr(resp, "data"):
            for item in resp.data:
                vectors.append(np.array(item.embedding, dtype=np.float32))
        else:
            vectors.append(np.array(resp["embedding"], dtype=np.float32))
        return vectors
    except Exception as e:
        # as fallback: try legacy client naming
        try:
            legacy = client.models.embed_content if hasattr(client.models, "embed_content") else client.embed
            resp = legacy(model=model, contents=texts)
            vectors = []
            if isinstance(resp, dict) and "data" in resp:
                for item in resp["data"]:
                    vectors.append(np.array(item["embedding"], dtype=np.float32))
            else:
                for item in resp:
                    vectors.append(np.array(item["embedding"], dtype=np.float32))
            return vectors
        except Exception as e2:
            raise RuntimeError(f"Embedding call failed: {e} / {e2}")

def build_faiss_index(vectors: List[np.ndarray]) -> faiss.IndexFlatIP:
    if len(vectors) == 0:
        raise ValueError("No vectors to build index.")
    dim = vectors[0].shape[0]
    index = faiss.IndexFlatIP(dim)  # inner-product index (works with normalized vectors)
    arr = np.vstack(vectors).astype('float32')
    # normalize for cosine similarity
    faiss.normalize_L2(arr)
    index.add(arr)
    return index

def search_index(index: faiss.IndexFlatIP, query_vec: np.ndarray, top_k: int = 4):
    q = query_vec.astype('float32').reshape(1, -1)
    faiss.normalize_L2(q)
    scores, idxs = index.search(q, top_k)
    return idxs[0], scores[0]

################################################################################
# App state
if "indexed" not in st.session_state:
    st.session_state.indexed = False
    st.session_state.text_chunks = []   # list of (source_name, chunk_text)
    st.session_state.index = None       # faiss index
    st.session_state.vectors = None     # numpy array
    st.session_state.client = None

# Sidebar: API key + model
st.sidebar.header("Settings & Keys")
api_key = st.sidebar.text_input("Google API key (GEMINI_API_KEY or GOOGLE_API_KEY)", type="password")
model_name = st.sidebar.text_input("Generative model (for answers)", value=os.environ.get("GEMINI_MODEL","gemini-1.5-pro"))
embed_model = st.sidebar.text_input("Embedding model", value=os.environ.get("GEMINI_EMBED","gemini-embedding-001"))
top_k = st.sidebar.slider("Top-k retrieved chunks", 1, 8, 4)

if api_key:
    os.environ["GEMINI_API_KEY"] = api_key
    os.environ["GOOGLE_API_KEY"] = api_key

# Upload docs
st.header("1) Upload documents to index")
uploaded = st.file_uploader("Upload .txt / .md / .pdf files (multiple)", accept_multiple_files=True)
if st.button("Load & Chunk"):
    if not uploaded:
        st.warning("Upload at least one file.")
    else:
        docs = load_documents(uploaded)
        splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = []
        for (name, text) in docs:
            # naive clean
            text = text.replace("\r\n", "\n")
            # prefer method name 'split_text' if present
            if hasattr(splitter, "split_text"):
                parts = splitter.split_text(text)
            else:
                # generic fallback (some versions use split_documents-like APIs)
                try:
                    parts = splitter.split(text)
                except Exception:
                    parts = [text]
            for i, part in enumerate(parts):
                if part.strip():
                    chunks.append((f"{name}__{i}", part.strip()))
        st.session_state.text_chunks = chunks
        st.session_state.indexed = False
        st.success(f"Loaded and split into {len(chunks)} chunks.")

# Build embeddings + FAISS
st.header("2) Build embeddings & FAISS index")
if st.button("Create Embeddings & Index"):
    if not st.session_state.text_chunks:
        st.warning("No chunks available — upload & chunk first.")
    else:
        try:
            client = make_genai_client()
            st.session_state.client = client
        except Exception as e:
            st.error("Could not initialize Gemini client: " + str(e))
            st.stop()

        texts = [t for (_, t) in st.session_state.text_chunks]
        with st.spinner("Calling Gemini to create embeddings (may cost API credits)..."):
            vectors = embed_texts(st.session_state.client, texts, model=embed_model)
        # convert to numpy array
        arr = np.vstack(vectors).astype('float32')
        faiss.normalize_L2(arr)
        idx = faiss.IndexFlatIP(arr.shape[1])
        idx.add(arr)
        st.session_state.index = idx
        st.session_state.vectors = arr
        st.session_state.indexed = True
        st.success(f"Indexed {len(texts)} chunks.")

# Query UI
st.header("3) Ask a question")
question = st.text_input("Enter your question here")
if st.button("Get Answer"):
    if not question:
        st.warning("Type a question.")
    elif not st.session_state.indexed:
        st.warning("Please create embeddings & index first.")
    else:
        client = st.session_state.client or make_genai_client()
        with st.spinner("Embedding query and retrieving..."):
            qvecs = embed_texts(client, [question], model=embed_model)
            qvec = qvecs[0]
            idxs, scores = search_index(st.session_state.index, qvec, top_k=top_k)
        retrieved_texts = []
        for i in idxs:
            try:
                src, chunk = st.session_state.text_chunks[int(i)]
                retrieved_texts.append((src, chunk))
            except Exception:
                pass

        # Build prompt: pass retrieved chunks as context
        context_text = "\n\n---\n\n".join([f"[{s}]\n{c}" for s, c in retrieved_texts])
        prompt = f"""You are an assistant. Use the following context from documents to answer the question. Be concise and cite the chunk names in square brackets when referring.

Context:
{context_text}

Question:
{question}

Answer:"""

        try:
            # Use the GenAI client to generate a response.
            # Modern SDK: client.models.generate_content(...)
            gen_resp = client.models.generate_content(model=model_name, prompt=prompt, max_output_tokens=512)
            # handle common shapes
            if isinstance(gen_resp, dict) and "candidates" in gen_resp:
                answer = gen_resp["candidates"][0]["content"]
            elif hasattr(gen_resp, "candidates"):
                answer = gen_resp.candidates[0].content
            else:
                # fallback: try reading 'output' or 'text'
                answer = str(gen_resp)
        except Exception as e:
            st.error("Generation call failed: " + str(e))
            answer = ""

        st.subheader("Answer")
        st.write(answer)

        if retrieved_texts:
            st.subheader("Retrieved passages")
            for src, txt in retrieved_texts:
                # Clean newlines before using inside f-string to avoid backslash-in-expression errors
                clean_txt = txt[:800].replace("\n", " ")
                st.markdown(f"**{src}** — {clean_txt}...")
