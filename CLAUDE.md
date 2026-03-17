# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an **Agentic RAG (Retrieval-Augmented Generation)** application that allows users to upload PDF documents and ask natural language questions. The system uses Google's Gemini models to provide intelligent, context-aware answers based on the uploaded document content.

**Key Features Implemented:**
- **Multi-Document Support**: Upload and query across multiple PDFs
- **Persistent Vector Store**: ChromaDB for document embeddings (survives restarts)
- **Session Management**: Server-side sessions with Flask-Session
- **Citation Support**: Answers include page numbers and source documents
- **React Frontend**: Modern UI with Vite (optional, falls back to vanilla HTML)

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
GOOGLE_API_KEY=your_google_gemini_api_key  # Alternative name, either works
GEMINI_MODEL=gemini-2.5-flash              # Optional, defaults to gemini-2.5-flash
GEMINI_EMBEDDING_MODEL=models/gemini-embedding-001  # Optional
FLASK_SECRET_KEY=your_secret_key_here        # Optional, auto-generated if not set
```

The app validates that `GEMINI_API_KEY` is set at startup and exits with an error if missing.

## Architecture

### High-Level Flow

1. **PDF Upload** (`POST /upload`): User uploads PDF → text extracted with page metadata → chunked with metadata → embedded → stored in ChromaDB
2. **Chat** (`POST /chat`): User asks question → Agent retrieves context with citations → Gemini generates answer with source references

### Module Structure

```
modules/
├── ingestion/              # Document processing
│   ├── pdf_reader.py       # pypdf-based text extraction with page metadata
│   └── chunker.py          # RecursiveCharacterTextSplitter with metadata preservation
├── retrieval/              # Vector search
│   ├── embedder.py         # Gemini embedding API with retry logic
│   ├── retriever.py        # ChromaDB wrapper with query/filter support
│   └── vector_store.py     # Persistent ChromaDB client
└── agent/                  # LLM agent
    └── agent.py            # LangChain tool-calling agent with Gemini

frontend/                   # React frontend (Vite)
├── src/
│   ├── components/         # React components
│   ├── App.jsx            # Main app component
│   └── main.jsx           # Entry point
└── package.json
```

### Key Architectural Patterns

**Agentic RAG**: Uses LangChain `create_tool_calling_agent` that decides *when* to call the retrieval tool. The agent can answer directly from its knowledge or retrieve context when needed.

**Session-Based State**: Each user session has:
- Unique session ID stored in Flask session cookie
- Isolated ChromaDB collection (`session_{session_id}`)
- Per-session agent instance cached in memory
- 24-hour session lifetime

**Persistent Vector Store**: ChromaDB stores embeddings on disk in `./chroma_db/`:
- Collections are named per session
- Documents can be added/removed individually
- Embeddings survive server restarts
- Supports metadata filtering by document_id

**Citation Tracking**: PDF processing preserves:
- Page numbers for each chunk
- Document name/source
- Chunk index within document
- Retrieved context includes citation info in responses

### API Endpoints

**Document Management:**
- `GET /api/session` - Get session info and document count
- `GET /api/documents` - List all documents in session
- `DELETE /api/documents/<id>` - Delete specific document
- `POST /api/documents/clear` - Remove all documents

**Upload & Chat:**
- `POST /upload` - Upload PDF (multipart/form-data)
- `POST /chat` - Ask question (JSON: `{question, include_sources}`)

**Frontend:**
- `GET /` - Serves React app (if built) or vanilla HTML template

### Error Handling

- The embedder has built-in retry logic for Google API rate limits (ResourceExhausted, ServiceUnavailable, etc.) with exponential backoff
- The agent has 3 retry attempts for transient failures
- Uploaded files are cleaned up after processing regardless of success/failure
- Session errors return appropriate HTTP status codes with error messages

### Data Flow

```
User Uploads PDF
    ↓
PDF Reader extracts text + page metadata
    ↓
Chunker splits text preserving page metadata
    ↓
Embedder generates embeddings (with retry)
    ↓
ChromaDB stores chunks + metadata in session collection
    ↓
Agent created with retriever tool
    ↓
User asks question
    ↓
Agent decides to retrieve context
    ↓
Retriever queries ChromaDB with cosine similarity
    ↓
Agent generates answer with citations
```
