# 🚀 Production Agentic RAG System (Gemini 1.5)

An industry-grade Retrieval-Augmented Generation (RAG) system built with **FastAPI**, **React**, and **Google Gemini 1.5**. This system features an autonomous agentic layer, hybrid retrieval (Dense + Sparse), and automated performance evaluation.

---

## 🏗️ Technical Architecture

The system is designed for high-performance and modularity:

*   **Agentic Orchestration**: Uses `LangChain` with Gemini 1.5 Flash to create an autonomous agent that decides when to retrieve context vs. answer from memory.
*   **Hybrid Retrieval Layer**:
    *   **Dense**: `ChromaDB` with `Gemini-1.5-Embeddings`.
    *   **Sparse**: `BM25` for keyword-based retrieval.
    *   **Parent-Child Chunking**: Maintains small chunks for retrieval but returns full parent context for generation.
*   **Reranking Pipeline**: Uses `BGE-Reranker` (Cross-Encoder) to prioritize the most relevant chunks before LLM generation.
*   **Evaluation Suite**: Built-in **RAGAS** integration to measure Faithfulness, Relevance, and Precision directly from the UI.
*   **Caching**: `Redis` integration for ultra-fast repeated query responses.

---

## 📂 Project Structure

```text
.
├── app/                    # Backend (FastAPI)
│   ├── api/                # REST endpoints
│   ├── core/               # App configuration & settings
│   ├── db/                 # Vector store & Redis logic
│   ├── ingestion/          # PDF/CSV/TXT/MD loaders
│   ├── models/             # Pydantic schemas
│   ├── retrieval/          # Hybrid search & Reranking
│   └── services/           # Business logic (Chat, Docs, Eval)
├── frontend/               # Frontend (React + Vite + Framer Motion)
│   ├── src/components/     # UI Components (Glassmorphism)
│   └── src/App.jsx         # Main Layout
├── data/                   # Persistent storage (Gitignored)
├── Dockerfile              # Multi-stage production build
└── pyproject.toml          # UV dependency management
```

---

## 🚀 Quick Start

### 1. Prerequisites
*   Python 3.12+
*   Node.js 20+
*   Gemini API Key (Google AI Studio)

### 2. Environment Setup
Create a `.env` file in the root:
```env
GEMINI_API_KEY=your_key
REDIS_URL=redis://localhost:6379/0  # Optional, fallbacks to in-memory
```

### 3. Local Development

**Backend:**
```bash
uv sync
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

### 4. Docker Deployment
```bash
docker build -t rag-prod .
docker run -p 8000:8000 -e GEMINI_API_KEY=your_key rag-prod
```

---

## ✨ Key Features

*   **Premium Glassmorphic UI**: High-end React interface with Framer Motion animations.
*   **Multi-Format Support**: Process PDF, CSV, TXT, and Markdown files.
*   **Autonomous Grounding**: LLM cites specific Source IDs to prevent hallucinations.
*   **Session Management**: Persistent local sessions with dedicated vector collections.
*   **Performance Analytics**: One-click RAGAS evaluation with visual metrics and celebration effects.
*   **Robust Caching**: Graceful Redis fallbacks for development environments.

---

## 🛠️ Built With
*   [FastAPI](https://fastapi.tiangolo.com/) - Backend Framework
*   [LangChain](https://www.langchain.com/) - LLM Orchestration
*   [ChromaDB](https://www.trychroma.com/) - Vector Database
*   [React](https://react.dev/) - Frontend Library
*   [Framer Motion](https://www.framer.com/motion/) - UI Animations
*   [RAGAS](https://ragas.io/) - Performance Evaluation
