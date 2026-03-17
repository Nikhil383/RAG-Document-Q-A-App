# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **Agentic RAG (Retrieval-Augmented Generation)** application that allows users to upload PDF documents and ask natural language questions. The system uses Google's Gemini models to provide intelligent, context-aware answers based on the uploaded document content.

**Key Features Implemented:**
- **Multi-Document Support**: Upload and query across multiple PDFs.
- **Persistent Vector Store**: ChromaDB for document embeddings (survives restarts).
- **Session Management**: Server-side sessions with Flask-Session.
- **Session Memory**: The agent remembers previous questions in the current session.
- **Batch Embedding**: Optimized document indexing with batch processing.
- **Citation Support**: Answers include page numbers and source documents.
- **React Frontend**: Modern UI with Vite (optional, falls back to vanilla HTML).

## Common Commands

### Running the Application

```bash
# Development (Flask dev server)
python app.py

# Production (Gunicorn)
gunicorn --bind 0.0.0.0:5000 app:app

# Development with React frontend
cd frontend
npm install
npm run dev  # Runs React dev server on :3000, proxies API to :5000
```

The Flask app will be available at `http://127.0.0.1:5000`.
The React dev server runs at `http://127.0.0.1:3000`.

### Building React Frontend

```bash
cd frontend
npm install
npm run build  # Builds to frontend/dist, served by Flask in production
```

### Docker

```bash
# Build and run with Docker
docker build -t rag-app .
docker run -p 5000:5000 --env-file .env rag-app
```

### Dependency Management

This project uses `uv` for dependency management:

```bash
# Install dependencies
uv sync

# Add a new dependency
uv add <package-name>

# Update lock file
uv lock
```

## Environment Variables

Create a `.env` file in the root directory with:

```env
GEMINI_API_KEY=your_google_gemini_api_key
GOOGLE_API_KEY=your_google_gemini_api_key  # Alternative name, used by LangChain
GEMINI_MODEL=gemini-1.5-flash                # Recommended
GEMINI_EMBEDDING_MODEL=models/gemini-embedding-001
FLASK_SECRET_KEY=your_secret_key_here        # Optional, auto-generated if not set
```

## Architecture

### High-Level Flow

1. **PDF Upload** (`POST /upload`): User uploads PDF → text extracted with page metadata → chunked with metadata → embedded in batches → stored in ChromaDB
2. **Chat** (`POST /chat`): User asks question → Agent retrieves context with citations → Gemini generates answer using memory and context

### Module Structure

```
modules/
├── ingestion/              # Document processing
│   ├── pdf_reader.py       # pypdf-based text extraction with page metadata
│   └── chunker.py          # RecursiveCharacterTextSplitter with metadata preservation
├── retrieval/              # Vector search
│   ├── embedder.py         # Gemini embedding API with batching and retry logic
│   ├── retriever.py        # ChromaDB wrapper with query/filter support
│   └── vector_store.py     # Persistent ChromaDB client
└── agent/                  # LLM agent
    └── agent.py            # LangChain tool-calling agent with Gemini & Memory
```

### Key Architectural Patterns

**Agentic RAG with Memory**: Uses LangChain `create_tool_calling_agent` with redundant memory via `ConversationBufferMemory`. The agent preserves state for follow-up questions within a session.

**Session-Based Isolation**: Each user session has:
- Isolated ChromaDB collection (`session_{session_id}`)
- Isolated Agent instance with its own memory
- Server-side stats tracking

**Efficient Retrieval**:
- Batch embedding generation for faster uploads.
- Cosine similarity search in ChromaDB.
- Metadata-preserved chunking for accurate citations.
