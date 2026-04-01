# RAG Web Q&A System

A Flask-based Retrieval-Augmented Generation (RAG) system for question-answering over uploaded documents.

## Features

- 📄 **Document Upload**: Upload PDF, CSV, TXT, and MD files
- 🔍 **Hybrid Retrieval**: Combines vector search (FAISS) with graph-based retrieval (Neo4j)
- 💬 **Q&A Interface**: Clean web UI for asking questions
- ⚡ **Streaming Responses**: Real-time answer streaming
- 🤖 **LangGraph Agent**: Orchestrates retrieval and generation

## Architecture

```
User Query → Hybrid Retriever → Context → LLM → Answer
                 ↓
         Vector Store (FAISS) + Graph Store (Neo4j)
```

## Prerequisites

- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- Neo4j database (local or cloud)
- OpenAI API key

## Installation

1. **Clone the repository**
   ```bash
   cd rag
   ```

2. **Install dependencies with uv**
   ```bash
   uv sync
   ```

3. **Set up environment variables**
   
   Create a `.env` file:
   ```env
   OPENAI_API_KEY=your-openai-api-key
   NEO4J_URI=bolt://localhost:7687
   NEO4J_USERNAME=your-neo4j-username
   NEO4J_PASSWORD=your-neo4j-password
   ```

4. **Start Neo4j**
   
   Make sure Neo4j is running on the configured URI.

## Usage

1. **Start the Flask server**
   ```bash
   uv run python app.py
   ```

2. **Open the web interface**
   
   Navigate to `http://localhost:5000`

3. **Upload documents**
   - Click "Choose a file" and select your document
   - Click "Upload & Index" to process the document

4. **Ask questions**
   - Type your question in the input field
   - Click "Ask" or press Enter
   - Watch the answer stream in real-time

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/upload` | POST | Upload and index a document |
| `/ask` | POST | Get answer (JSON response) |
| `/ask_stream` | POST | Get answer (streaming) |
| `/docs` | GET | List uploaded documents |
| `/health` | GET | Health check |

## Project Structure

```
rag/
├── app.py                      # Flask application
├── langraph_agent.py           # LangGraph agent definition
├── modules/
│   ├── graph/
│   │   └── neo4j_store.py      # Neo4j graph database interface
│   ├── ingestion/
│   │   ├── chunker.py          # Document chunking logic
│   │   ├── graph_builder.py    # Knowledge graph builder
│   │   └── loader.py           # Document loaders
│   ├── llm/
│   │   └── llm.py              # LLM configuration
│   └── retriever/
│       ├── graph_retriever.py  # Graph-based retrieval
│       ├── hybrid_retriever.py # Hybrid retrieval (vector + graph)
│       └── vector_store.py     # FAISS vector store
├── templates/
│   └── index.html              # Web UI
├── data/
│   └── uploads/                # Uploaded documents
├── pyproject.toml              # Project dependencies
└── README.md
```

## How It Works

1. **Document Ingestion**
   - Documents are loaded and split into chunks
   - Chunks are indexed in FAISS for vector similarity search
   - Chunks are also stored in Neo4j as a knowledge graph

2. **Query Processing**
   - User query is sent to the hybrid retriever
   - Retriever searches both vector store and graph
   - Results are combined and deduplicated

3. **Answer Generation**
   - Retrieved context is sent to the LLM
   - LLM generates an answer based on the context
   - Answer is streamed back to the user

## Customization

### Adjust Chunk Size

Edit `modules/ingestion/chunker.py`:
```python
chunk_docs(docs, chunk_size=500, chunk_overlap=50)
```

### Change LLM Model

Edit `modules/llm/llm.py`:
```python
return ChatOpenAI(
    model="gpt-4o",  # Change model here
    temperature=0
)
```

### Modify Retrieval Strategy

Edit `modules/retriever/hybrid_retriever.py` to customize how results are combined.

## Troubleshooting

**Neo4j Connection Error**
- Ensure Neo4j is running
- Check credentials in `.env`
- Verify the URI is correct

**No Results from Retrieval**
- Make sure documents are uploaded and indexed
- Try different query phrasings
- Check if chunks were created properly

**OpenAI API Error**
- Verify your API key is valid
- Check your API quota

## License

MIT
