import os
import tempfile
from pathlib import Path
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename

from src.multimodal_rag import MultimodalRAG
from src.multimodal_rag.config import CONFIG

UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
ALLOWED_EXTENSIONS = {"pdf", "txt", "png", "jpg", "jpeg", "webp", "bmp", "gif"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["SECRET_KEY"] = os.environ.get("FLASK_SECRET_KEY", "change-this-secret")

rag = MultimodalRAG()


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def index():
    answer = None
    sources = None
    if request.method == "POST":
        if "query" in request.form and request.form["query"].strip():
            query = request.form["query"].strip()
            result = rag.query(query)
            answer = result["answer"]
            sources = result["sources"]
        elif "file" in request.files:
            file = request.files["file"]
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                upload_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(upload_path)
                ext = filename.rsplit(".", 1)[1].lower()
                try:
                    if ext == "pdf":
                        rag.ingest_pdf(upload_path)
                    elif ext == "txt":
                        rag.ingest_text_file(upload_path)
                    else:
                        rag.ingest_image_file(upload_path)
                    flash(f"Uploaded and indexed {filename}", "success")
                except Exception as exc:
                    flash(str(exc), "danger")
            else:
                flash("No valid file selected. Allowed types: PDF, TXT, PNG, JPG, JPEG, WEBP, BMP, GIF.", "warning")
        return redirect(url_for("index"))

    return render_template("index.html", answer=answer, sources=sources, total_indexed=rag.stats()["total_indexed"])


@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


if __name__ == "__main__":
    app.run(debug=True)
