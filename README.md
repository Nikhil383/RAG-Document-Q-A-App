# 🧠 Retrieval-Augmented Generation (RAG) Document Q&A Web App

An end-to-end **Flask-based AI web application** that allows users to **upload documents (PDF, DOCX, or TXT)** and ask **contextual questions** about their contents using **semantic search** and **text generation** via advanced NLP models.

---

## 🚀 Features

✅ Upload & process documents in **PDF, DOCX, or TXT** format  
✅ Uses **LangChain**, **FAISS**, and **SentenceTransformers** for semantic retrieval  
✅ Generates contextual answers using **FLAN-T5** or other Hugging Face models  
✅ Responsive, modern **HTML/CSS/JS UI** with upload, remove, and query functions  
✅ Modular backend with clean architecture and reusability  
✅ Works seamlessly on **CPU** — no GPU required  

---

## 🧩 Tech Stack

**Frontend:** HTML5, CSS3, JavaScript  
**Backend:** Flask (Python)  
**AI/NLP:** LangChain, SentenceTransformers, FAISS, Hugging Face Transformers, PyTorch  
**Document Parsing:** PyPDF2, python-docx  
**Deployment:** Docker / Render / Hugging Face Spaces  

---

## 📁 Project Structure

rag_app/
│
├── app.py # Flask entry point
├── requirements.txt # Dependencies
│
├── backend/
│ ├── init.py
│ └── rag_engine.py # Core RAG logic (FAISS + Hugging Face)
│
├── uploads/ # Stores user-uploaded files
│
├── templates/
│ └── index.html # Frontend HTML UI
│
└── static/
└── style.css # Custom styles



---

## ⚙️ Setup Instructions

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/<your-username>/rag-document-qa.git
cd rag-document-qa

2️⃣ Create Virtual Environment
python -m venv venv
venv\Scripts\activate   # Windows
# OR
source venv/bin/activate   # macOS/Linux

3️⃣ Install Requirements
pip install -r requirements.txt

4️⃣ Run the App
python app.py


App will be available at http://127.0.0.1:5000

🧠 How It Works

1️⃣ User Uploads Document

Accepts .pdf, .docx, and .txt files.

Extracts raw text using PyPDF2 or python-docx.

2️⃣ Text Processing & Indexing

Splits text into chunks using LangChain’s RecursiveCharacterTextSplitter.

Generates embeddings via SentenceTransformer.

Builds FAISS index for fast semantic retrieval.

3️⃣ Question Answering

Retrieves top relevant text chunks.

Generates context-aware answers using google/flan-t5-base.

🤖 Model Options

You can switch models in backend/rag_engine.py:

# Default (text generation)
self.generator = pipeline("text2text-generation", model="google/flan-t5-base")

# Alternative (extractive QA)
# self.generator = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Advanced (GPU)
# self.generator = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.2", device_map="auto")

🎨 User Interface Preview
Upload Document	Ask Questions	Get Answers

	
	

(Add screenshots in a screenshots/ folder after running the app.)

📦 requirements.txt
flask
werkzeug
transformers
sentence-transformers
faiss-cpu
langchain
langchain-core
langchain-text-splitters
PyPDF2
python-docx
torch
numpy


💡 Future Enhancements

 Support for multiple document uploads

 Persistent FAISS index storage

 Model selection toggle in UI

 Dark mode 🌙

 Deploy to Hugging Face Spaces or Render


 🧠 Author

👤 Your Name
📧 your.email@example.com

🌐 GitHub
 | LinkedIn


 🪪 License

This project is licensed under the MIT License — free to use, modify, and distribute.