# Agentic RAG Document Q&A App

## Problem Statement
In the era of information overload, manually searching through large PDF documents for specific answers is time-consuming and inefficient. Traditional keyword searches often fail to capture the context of user queries, leading to irrelevant results. There is a need for an intelligent system that can understand natural language questions, retrieve relevant context from documents, and generate accurate, concise answers.

## Project Overview
This project is an **Agentic Retrieval-Augmented Generation (RAG)** application designed to solve this problem. It leverages **Google's Gemini 3 Flash Preview** model as an intelligent agent capable of reasoning and using tools. The application allows users to upload PDF documents, which are then processed and indexed. Users can then ask natural language questions, and the agent dynamically retrieves relevant information from the document to provide precise answers.

The application features a clean, responsive web interface built with **Flask** and **HTML/CSS**, making it accessible and easy to use.

## Project Structure
The project is organized into modular components for maintainability and scalability:

```
RAG/
├── app.py                  # Main Flask application entry point
├── modules/                # Core logic modules
│   ├── agent.py            # Gemini Agent configuration and tool definitions
│   ├── chunker.py          # Text splitting/chunking logic
│   ├── embedder.py         # Embedding generation using Gemini
│   ├── pdf_reader.py       # PDF text extraction
│   └── retriever.py        # Vector search implementation (FAISS)
├── templates/
│   └── index.html          # Frontend user interface
├── uploads/                # Directory for storing uploaded PDFs
├── data/                   # Directory for additional data storage
├── pyproject.toml          # Project dependencies (UV/Pip)
└── .env                    # Environment variables (API Keys)
```

## Steps to Execute the Project

### Prerequisites
- Python 3.11 or higher
- A Google Cloud Project with the **Gemini API** enabled and an API Key.

### Installation

1.  **Clone the Repository**
    ```bash
    git clone <repository_url>
    cd RAG
    ```

2.  **Create a Virtual Environment** (Recommended)
    ```bash
    python -m venv .venv
    # Windows
    .venv\Scripts\activate
    # macOS/Linux
    source .venv/bin/activate
    ```

3.  **Install Dependencies**
    You can install the required packages using pip:
    ```bash
    pip install flask google-generativeai pypdf python-dotenv faiss-cpu
    ```
    *Note: The project also supports management via `uv` if preferred.*

4.  **Set Up Environment Variables**
    Create a `.env` file in the root directory and add your Google Gemini API Key:
    ```env
    GEMINI_API_KEY=your_api_key_here
    ```

### Running the Application

1.  Start the Flask server:
    ```bash
    python app.py
    ```

2.  Open your web browser and navigate to:
    ```
    http://127.0.0.1:5000
    ```

3.  **Usage**:
    - Click "Choose File" to upload a PDF document.
    - Wait for the "Successfully processed" message.
    - Type your question in the chat box and press "Send".

## Architecture for Agentic RAG

The system follows an **Agentic RAG** architecture, which differs from standard RAG by giving the LLM autonomy to decide *when* and *how* to retrieve information.

1.  **Ingestion Layer**:
    - **PDF Reader**: Extracts raw text from uploaded PDF files (`pypdf`).
    - **Chunking**: Splits text into semantically meaningful chunks to optimize retrieval accuracy.
    - **Embedding**: Converts text chunks into high-dimensional vector embeddings using Google's `text-embedding-004` (or similar) model.

2.  **Storage Layer**:
    - **Vector Store**: Uses **FAISS (Facebook AI Similarity Search)** to store embeddings in memory for fast similarity search.

3.  **Agentic Layer**:
    - **Gemini Agent**: The core of the system is a `GenerativeModel` (Gemini 3 Flash Preview) configured with tools.
    - **Tool Use**: The retrieval logic is exposed as a function (`retriever_tool`). The agent analyzes the user's query and decides if it needs to call this tool to fetch external knowledge from the PDF.
    - **Generation**: The agent synthesizes the retrieved data with its internal knowledge to answer the user's question contextually.

4.  **Application Layer**:
    - **Flask Backend**: Manages API endpoints (`/upload`, `/chat`) and orchestrates the flow.
    - **Frontend**: A simple client-side application to handle file uploads and chat interactions.

## Improvements (In Future)

To make this application production-ready, multiple improvements are planned:

-   **Multi-Document Support**: Enable querying across multiple uploaded files simultaneously.
-   **Vector Database Persistence**: Switch from in-memory FAISS to a persistent vector store (pinecone/chromadb) to save embeddings effectively.
-   **Session Management**: Implement user sessions so multiple users can use the app concurrently without state collisions.
-   **Advanced Chunking**: Implement semantic chunking or recursive character splitting for better context preservation.
-   **UI Enhancement**: Migrate the frontend to a more robust framework like **React** for a richer user experience.
-   **Citation Support**: Modify the agent to provide specific page numbers or source quotes for its answers.
