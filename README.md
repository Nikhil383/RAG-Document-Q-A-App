# Graph RAG Document Q&A App

A Graph-based Retrieval-Augmented Generation (RAG) application using:
- **Flask** backend API
- **React + Vite** frontend
- **Gemini 2.0 Flash** for extraction and answering
- **Neo4j** as the knowledge graph store

## Current Architecture

```text
.
├── app.py                # Flask API server (`/api/upload`, `/api/chat`)
├── ingest.py             # Document parsing + graph ingestion into Neo4j
├── graph_agent.py        # LangGraph workflow for entity extraction + answer generation
├── frontend/             # React app (Vite)
├── templates/            # Static/template assets
├── Dockerfile            # Multi-stage build for frontend + backend runtime
└── pyproject.toml        # Python dependencies
```

## Prerequisites
- Python 3.11+
- Node.js 20+
- Running Neo4j instance (default `bolt://localhost:7687`)
- Gemini API key

## Environment
Create `.env` in repository root:

```env
GEMINI_API_KEY=your_key
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password
```

## Local Development

### Backend
```bash
uv sync
uv run python app.py
```
Backend runs on `http://127.0.0.1:5000`.

### Frontend
```bash
cd frontend
npm install
npm run dev
```
Frontend runs on `http://127.0.0.1:3000` and proxies `/api` to Flask.

## Docker
```bash
docker build -t graph-rag-app .
docker run --rm -p 5000:5000 --env-file .env graph-rag-app
```

## API Endpoints

### `POST /api/upload`
Upload one `.txt`, `.md`, `.csv`, or `.pdf` document and ingest it into Neo4j.

### `POST /api/chat`
Request body:
```json
{ "query": "Your question" }
```
Returns answer, extracted entities, and context used.

## Notes
- Ingestion depends on external Gemini + Neo4j availability.
- PDF support requires `pypdf` (included in project dependencies).
