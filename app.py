import os
from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename

# Import our previous graph scripts
from ingest import ingest_data
from graph_agent import build_agent

ALLOWED_EXTENSIONS = {"txt", "md", "csv", "pdf"}
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB

app = Flask(__name__)
# Enable CORS for React dev server
CORS(app)

app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


# Instantiate the Agent globally to avoid recompiling the graph diagram for every query
print("Initializing the LangGraph Agent...")
agent_executor = build_agent()


def _allowed_file(filename: str) -> bool:
    if "." not in filename:
        return False
    return filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.errorhandler(413)
def request_entity_too_large(_error):
    return jsonify({"error": f"File too large. Max size is {MAX_UPLOAD_BYTES // (1024 * 1024)}MB."}), 413


@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file found in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    filename = secure_filename(file.filename)
    if not _allowed_file(filename):
        return jsonify({
            "error": "Unsupported file type. Allowed types: .txt, .md, .csv, .pdf"
        }), 400

    save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(save_path)

    try:
        # Call the standalone ingest logic directly on the uploaded file
        ingest_data(save_path)
        return jsonify({'message': f'File {filename} successfully parsed into Neo4j Knowledge Graph.'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        # Clean up the temp file after ingestion
        if os.path.exists(save_path):
            os.remove(save_path)


@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    if not data or 'query' not in data:
        return jsonify({'error': 'Query is required'}), 400

    query_text = str(data['query']).strip()
    if not query_text:
        return jsonify({'error': 'Query cannot be empty'}), 400

    # Run the user's query through the Agent workflow
    inputs = {
        "query": query_text,
        "refine_count": 0
    }

    try:
        result = agent_executor.invoke(inputs)
        return jsonify({
            'answer': result['answer'],
            'entities_searched': result.get('entities', []),
            'context_used': result.get('context', '')
        })
    except Exception as e:
        return jsonify({'error': f'Agent execution failed: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)
