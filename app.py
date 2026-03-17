import os
import sys
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, Optional
from threading import Lock

from flask import Flask, render_template, request, jsonify, session, send_from_directory
from flask_session import Session
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# Check if React build exists
REACT_BUILD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frontend', 'dist')
USE_REACT = os.path.exists(REACT_BUILD_DIR)

from modules import (
    read_pdf_with_metadata,
    chunk_text_with_metadata,
    get_embedding,
    Retriever,
    Agent,
    extract_pdf_info
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Validate that the API key is present before starting the app
if not os.getenv("GEMINI_API_KEY"):
    logger.error("GEMINI_API_KEY environment variable is not set. Please set it in your .env file.")
    sys.exit(1)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('FLASK_SECRET_KEY', os.urandom(32))
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Configure server-side sessions
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_FILE_DIR'] = './.flask_session'
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(hours=24)
Session(app)

# Ensure upload and session directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)

# Session storage for agents (in-memory with basic cleanup)
# In production, use Redis or a database
session_agents: Dict[str, Agent] = {}
session_lock = Lock()


def get_or_create_session_id() -> str:
    """Get existing session ID or create a new one."""
    if 'session_id' not in session:
        session['session_id'] = str(uuid.uuid4())
        session['created_at'] = datetime.now().isoformat()
        session.permanent = True
    return session['session_id']


def get_agent_for_session(session_id: str) -> Optional[Agent]:
    """Get the agent for a session, creating it if needed."""
    with session_lock:
        if session_id not in session_agents:
            # Check if session has documents
            retriever = Retriever(session_id=session_id)
            stats = retriever.get_stats()

            if stats.get('document_count', 0) > 0:
                # Create agent for existing documents
                agent = Agent(retriever=retriever, session_id=session_id)
                session_agents[session_id] = agent

        return session_agents.get(session_id)


def set_agent_for_session(session_id: str, agent: Agent):
    """Set the agent for a session."""
    with session_lock:
        session_agents[session_id] = agent


def remove_agent_for_session(session_id: str):
    """Remove the agent for a session."""
    with session_lock:
        if session_id in session_agents:
            del session_agents[session_id]


@app.route('/')
def index():
    """Render the main page."""
    # Ensure session exists
    get_or_create_session_id()

    if USE_REACT:
        return send_from_directory(REACT_BUILD_DIR, 'index.html')
    return render_template('index.html')


@app.route('/<path:filename>')
def serve_static(filename):
    """Serve static files from React build."""
    if USE_REACT:
        file_path = os.path.join(REACT_BUILD_DIR, filename)
        if os.path.exists(file_path):
            return send_from_directory(REACT_BUILD_DIR, filename)
    return jsonify({'error': 'Not found'}), 404


@app.route('/api/documents', methods=['GET'])
def list_documents():
    """List all documents in the current session."""
    session_id = get_or_create_session_id()

    try:
        retriever = Retriever(session_id=session_id)
        documents = retriever.get_documents()

        return jsonify({
            'documents': documents,
            'session_id': session_id
        }), 200
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/documents/<doc_id>', methods=['DELETE'])
def delete_document(doc_id: str):
    """Delete a specific document from the session."""
    session_id = get_or_create_session_id()

    try:
        retriever = Retriever(session_id=session_id)
        success = retriever.delete_document(doc_id)

        if success:
            # Remove agent to force recreation with updated documents
            remove_agent_for_session(session_id)
            return jsonify({'message': f'Document {doc_id} deleted successfully'}), 200
        else:
            return jsonify({'error': 'Document not found or could not be deleted'}), 404
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/documents/clear', methods=['POST'])
def clear_all_documents():
    """Clear all documents from the current session."""
    session_id = get_or_create_session_id()

    try:
        from modules.retrieval.vector_store import get_vector_store
        vector_store = get_vector_store()
        success = vector_store.delete_collection(session_id)

        # Remove agent
        remove_agent_for_session(session_id)

        if success:
            return jsonify({'message': 'All documents cleared successfully'}), 200
        else:
            return jsonify({'message': 'No documents to clear'}), 200
    except Exception as e:
        logger.error(f"Error clearing documents: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle PDF upload and processing."""
    session_id = get_or_create_session_id()
    filepath = None

    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not file or not file.filename.endswith('.pdf'):
        return jsonify({'error': 'Invalid file type. Only PDFs are supported.'}), 400

    filename = secure_filename(file.filename)
    document_id = str(uuid.uuid4())
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    try:
        # Save the file temporarily
        file.save(filepath)

        # Extract PDF info
        pdf_info = extract_pdf_info(filepath)
        logger.info(f"Processing PDF: {filename} ({pdf_info['num_pages']} pages)")

        # Process PDF with metadata
        pages = read_pdf_with_metadata(filepath, document_name=filename)

        if not pages:
            return jsonify({'error': 'Could not extract text from PDF'}), 400

        # Chunk with metadata preservation
        chunks_with_metadata = chunk_text_with_metadata(
            pages,
            chunk_size=1000,
            chunk_overlap=200,
            document_id=document_id
        )

        if not chunks_with_metadata:
            return jsonify({'error': 'No text content found in PDF'}), 400

        # Extract chunks and metadata
        chunks = [c['text'] for c in chunks_with_metadata]
        metadata_list = [c['metadata'] for c in chunks_with_metadata]

        # Add to vector store
        retriever = Retriever(session_id=session_id)
        chunk_ids = retriever.add_documents(
            chunks=chunks,
            metadata=metadata_list,
            document_id=document_id
        )

        # Create or update agent
        agent = Agent(retriever=retriever, session_id=session_id)
        set_agent_for_session(session_id, agent)

        # Get updated document list
        documents = retriever.get_documents()

        return jsonify({
            'message': f'Successfully processed {filename} with {len(chunks)} chunks.',
            'document_id': document_id,
            'document_name': filename,
            'pages': pdf_info['num_pages'],
            'chunks': len(chunks),
            'total_documents': len(documents)
        }), 200

    except Exception as e:
        logger.error(f"Error processing upload: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

    finally:
        # Clean up uploaded file
        if filepath and os.path.exists(filepath):
            try:
                os.remove(filepath)
            except OSError as e:
                logger.error(f"Failed to clean up uploaded file {filepath}: {e}")


@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat messages."""
    session_id = get_or_create_session_id()

    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    question = data.get('question')
    include_sources = data.get('include_sources', False)

    if not question:
        return jsonify({'error': 'No question provided'}), 400

    try:
        # Get or create agent for this session
        agent = get_agent_for_session(session_id)

        if not agent:
            return jsonify({
                'error': 'Please upload a PDF first.',
                'code': 'NO_DOCUMENTS'
            }), 400

        if include_sources:
            result = agent.ask_with_sources(question)
            return jsonify({
                'answer': result['answer'],
                'sources': [
                    {
                        'text': s['text'][:500] + '...' if len(s['text']) > 500 else s['text'],
                        'metadata': s.get('metadata', {}),
                        'distance': s.get('distance', 0)
                    }
                    for s in result['sources']
                ]
            })
        else:
            answer = agent.ask(question)
            return jsonify({'answer': answer})

    except Exception as e:
        logger.error(f"Error in chat: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500


@app.route('/api/session', methods=['GET'])
def get_session_info():
    """Get current session information."""
    session_id = get_or_create_session_id()

    try:
        retriever = Retriever(session_id=session_id)
        stats = retriever.get_stats()
        documents = retriever.get_documents()

        return jsonify({
            'session_id': session_id,
            'created_at': session.get('created_at'),
            'document_count': stats.get('document_count', 0),
            'documents': documents
        }), 200
    except Exception as e:
        logger.error(f"Error getting session info: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
