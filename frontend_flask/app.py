from flask import Flask, render_template, request, jsonify
import requests

app = Flask(__name__)
FASTAPI_URL = "http://localhost:8000/api/v1"

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def proxy_chat():
    try:
        data = request.json
        resp = requests.post(f"{FASTAPI_URL}/chat", json=data)
        return jsonify(resp.json()), resp.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/upload", methods=["POST"])
def proxy_upload():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file uploaded"}), 400
        
        file = request.files['file']
        files = {'file': (file.filename, file.stream, file.mimetype)}
        resp = requests.post(f"{FASTAPI_URL}/upload", files=files)
        return jsonify(resp.json()), resp.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/evaluate", methods=["POST"])
def proxy_evaluate():
    try:
        data = request.json
        resp = requests.post(f"{FASTAPI_URL}/evaluate", json=data)
        return jsonify(resp.json()), resp.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5000, debug=True)
