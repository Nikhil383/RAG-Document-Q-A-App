# app.py
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename
import os
from backend.rag_engine import RAGPipeline

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"pdf", "txt", "docx"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

rag = RAGPipeline()


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        file.save(filepath)
        rag.load_document(filepath)
        return jsonify({"message": f"File '{filename}' uploaded and processed successfully."})
    else:
        return jsonify({"error": "Unsupported file type. Use PDF, TXT, or DOCX."}), 400


@app.route("/ask", methods=["POST"])
def ask():
    query = request.form.get("query")
    answer = rag.answer(query)
    return jsonify({"answer": answer})


if __name__ == "__main__":
    app.run(debug=True)
