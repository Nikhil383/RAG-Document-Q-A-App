import os
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from modules.pdf_reader import read_pdf
from modules.chunker import chunk_text
from modules.embedder import get_embedding
from modules.retriever import Retriever
from modules.agent import Agent

load_dotenv()

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload

# Global agent instance (for single-user simplicity in this demo)
agent = None

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    global agent
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file and file.filename.endswith('.pdf'):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Process PDF
            text = read_pdf(filepath)
            chunks = chunk_text(text)
            embeddings = [get_embedding(chunk) for chunk in chunks]
            retriever = Retriever(chunks, embeddings)
            agent = Agent(retriever)
            
            return jsonify({'message': f'Successfully processed {filename} with {len(chunks)} chunks.'}), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Invalid file type'}), 400

@app.route('/chat', methods=['POST'])
def chat():
    global agent
    if not agent:
        return jsonify({'error': 'Please upload a PDF first.'}), 400
    
    data = request.get_json()
    question = data.get('question')
    
    if not question:
        return jsonify({'error': 'No question provided'}), 400
    
    try:
        answer = agent.ask(question)
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
