# Agentic RAG Document Q&A App

## Problem Statement
In the era of information overload, manually searching through large PDF documents for specific answers is time-consuming and inefficient. Traditional keyword searches often fail to capture the context of user queries, leading to irrelevant results. There is a need for an intelligent system that can understand natural language questions, retrieve relevant context from documents, and generate accurate, concise answers.

## Project Overview
This project is an **Agentic Retrieval-Augmented Generation (RAG)** application designed to solve this problem. It leverages **Google's Gemini 1.5 Flash** model as an intelligent agent capable of reasoning and using tools. The application allows users to upload PDF documents, which are then processed and indexed. Users can then ask natural language questions, and the agent dynamically retrieves relevant information from the document to provide precise answers with citations.

The application features a modern **React** frontend and a robust **Flask** backend with persistent vector storage.

## Project Structure
The project is organized into modular components for maintainability and scalability:

```
RAG/
├── app.py                  # Main Flask application entry point
├── modules/                # Core logic modules
│   ├── agent/              # Gemini Agent with tool-use and memory
│   ├── ingestion/          # PDF processing and text chunking
│   └── retrieval/          # Embedding (Gemini) and vector search (ChromaDB)
├── frontend/               # React (Vite) frontend
├── templates/              # Fallback HTML interface
├── uploads/                # Directory for temporary PDF storage
├── chroma_db/              # Persistent vector database
├── pyproject.toml          # Project dependencies (UV)
└── .env                    # Environment variables (API Keys)
```

## Steps to Execute the Project

### Prerequisites
- Python 3.11 or higher
- Node.js & npm (for React frontend)
- A Google Gemini API Key.

### Installation

1.  **Clone the Repository**
    ```bash
    git clone <repository_url>
    cd RAG_Document
    ```

2.  **Set Up Python Environment**
    ```bash
    uv sync
    # or
    python -m venv .venv
    source .venv/bin/activate  # .venv\Scripts\activate on Windows
    pip install -e .
    ```

3.  **Set Up Environment Variables**
    Create a `.env` file in the root directory:
    ```env
    GEMINI_API_KEY=your_api_key_here
    GOOGLE_API_KEY=your_api_key_here
    ```

### Running the Application

1.  Start the Flask server:
    ```bash
    python app.py
    ```

2.  (Optional) Start the React Dev server:
    ```bash
    cd frontend
    npm install
    npm run dev
    ```

3.  Open `http://127.0.0.1:5000` (or `:3000` if using React dev server).

## Architecture for Agentic RAG

The system follows an **Agentic RAG** architecture:

1.  **Ingestion Layer**:
    - **PDF Reader**: Extracts text and page metadata using `pypdf`.
    - **Chunking**: Splits text into chunks while preserving metadata for citations.
    - **Embedding**: Generates vectors using Google's `text-embedding-004` (via batch API).

2.  **Storage Layer**:
    - **Vector Store**: Uses **ChromaDB** for persistent, local storage of embeddings.

3.  **Agentic Layer**:
    - **Gemini Agent**: Configured with `langchain` tools and `ConversationBufferMemory`.
    - **Autonomy**: The agent decides when to call the `retrieve_context` tool vs answering from memory.

4.  **Application Layer**:
    - **Flask Backend**: Manages sessions, document uploads, and agent lifecycle.
    - **React Frontend**: Provides a premium, responsive glassmorphism UI.

## Key Features
- **Persistent Memory**: Remembers chat history for follow-up questions.
- **Accurate Citations**: References specific page numbers and source documents.
- **Batch Processing**: Fast document indexing.
- **Session Isolation**: Each user gets their own dedicated agent and collection.
- **React UI**: Modern look with shadcn/ui inspired components.
