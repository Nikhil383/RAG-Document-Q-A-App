from flask import Flask, request, jsonify, render_template, Response
import os
import uuid
from datetime import datetime

from modules.retriever.vector_store import VectorStore
from modules.graph.neo4j_store import Neo4jStore
from modules.retriever.graph_retriever import GraphRetriever
from modules.retriever.hybrid_retriever import HybridRetriever
from modules.ingestion.chunker import chunk_docs
from modules.ingestion.loader import load_document
from modules.ingestion.graph_builder import GraphBuilder
from langraph_agent import build_graph

app = Flask(__name__)

UPLOAD_FOLDER = "data/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Lazy initialization - components created on first use
_vector_store = None
_graph_store = None
_graph_builder = None
_graph_retriever = None
_hybrid_retriever = None
_agent = None

# Store for tracking uploaded documents
uploaded_docs = {}


def get_components():
    """Initialize and return all components lazily"""
    global _vector_store, _graph_store, _graph_builder, _graph_retriever, _hybrid_retriever, _agent
    
    if _vector_store is None:
        _vector_store = VectorStore()
        _graph_store = Neo4jStore(
            os.getenv("NEO4J_URI"),
            os.getenv("NEO4J_USERNAME"),
            os.getenv("NEO4J_PASSWORD")
        )
        _graph_builder = GraphBuilder(
            os.getenv("NEO4J_URI"),
            os.getenv("NEO4J_USERNAME"),
            os.getenv("NEO4J_PASSWORD")
        )
        _graph_retriever = GraphRetriever(_graph_store)
        _hybrid_retriever = HybridRetriever(_vector_store, _graph_retriever)
        _agent = build_graph(_hybrid_retriever)
    
    return _vector_store, _graph_store, _graph_builder, _graph_retriever, _hybrid_retriever, _agent


def get_agent():
    """Get the agent instance"""
    get_components()  # Ensure initialized
    return _agent


def get_hybrid_retriever():
    """Get the hybrid retriever instance"""
    get_components()  # Ensure initialized
    return _hybrid_retriever


def get_graph_builder():
    """Get the graph builder instance"""
    get_components()  # Ensure initialized
    return _graph_builder


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    """Upload and index a document"""
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400
    
    # Generate unique ID for this document
    doc_id = str(uuid.uuid4())[:8]
    filename = f"{doc_id}_{file.filename}"
    path = os.path.join(UPLOAD_FOLDER, filename)
    
    file.save(path)
    
    try:
        # Load document
        docs = load_document(path)
        
        # Chunk the document
        chunks = chunk_docs(docs)
        
        # Get components
        vector_store = get_hybrid_retriever().vector_store
        graph_builder = get_graph_builder()
        
        # Index in vector store
        vector_store.build(chunks)
        
        # Build knowledge graph
        graph_builder.create_graph(chunks)
        
        # Track uploaded document
        uploaded_docs[doc_id] = {
            "filename": file.filename,
            "chunks": len(chunks),
            "uploaded_at": datetime.now().isoformat()
        }
        
        return jsonify({
            "status": "success",
            "message": f"Uploaded & indexed {len(chunks)} chunks",
            "doc_id": doc_id,
            "filename": file.filename
        })
    
    except Exception as e:
        # Clean up on error
        if os.path.exists(path):
            os.remove(path)
        return jsonify({"error": str(e)}), 500


@app.route("/ask", methods=["POST"])
def ask():
    """Get answer to a question"""
    data = request.json
    query = data.get("query", "")
    
    if not query:
        return jsonify({"error": "Query is required"}), 400
    
    try:
        agent = get_agent()
        result = agent.invoke({"query": query})
        return jsonify({
            "answer": result["answer"],
            "query": query
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/ask_stream", methods=["POST"])
def ask_stream():
    """Stream answer to a question"""
    data = request.json
    query = data.get("query", "")
    
    if not query:
        return jsonify({"error": "Query is required"}), 400
    
    def generate():
        try:
            agent = get_agent()
            result = agent.invoke({"query": query})
            answer = result.get("answer", "")
            for word in answer.split():
                yield word + " "
        except Exception as e:
            yield f"Error: {str(e)}"
    
    return Response(generate(), mimetype="text/plain")


@app.route("/docs", methods=["GET"])
def list_docs():
    """List uploaded documents"""
    return jsonify({"documents": list(uploaded_docs.values())})


@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint"""
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    app.run(debug=True, port=5000)
