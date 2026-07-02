# Multimodal RAG System

A retrieval-augmented generation pipeline that indexes **both text and images**
into a shared embedding space, retrieves the most relevant items of either
type for a query, and generates an answer that's actually grounded in the
retrieved images (not just their captions).

## How it works

```
                ┌─────────────┐
   PDF/images → │  Document    │  → text chunks + extracted images
                │  Processor   │
                └──────┬───────┘
                       │
                       ▼
                ┌──────────────┐
                │  CLIP         │  text and images → same vector space
                │  Embedder     │
                └──────┬───────┘
                       │
                       ▼
                ┌──────────────┐
                │  ChromaDB     │  persisted vector store
                │  Vector Store │
                └──────┬───────┘
                       │  query
                       ▼
   question → ┌──────────────┐      top-k text chunks
              │  Retriever    │  →   top-k images
              └──────┬───────┘
                     │
                     ▼
              ┌──────────────┐
              │  Generator    │  Claude, given text + REAL images
              │  (Claude)     │  as vision input → grounded answer
              └──────────────┘
```

**Key design choice:** CLIP embeds text and images into the *same* vector
space, so a text question can directly retrieve relevant images without any
intermediate captioning step. Text and image hits are ranked *separately*
(not merged into one list) because their CLIP similarity scores aren't
directly comparable across modalities — merging them tends to silently
starve one modality.

At generation time, retrieved images are sent to Claude as actual base64
image blocks, so the model reasons over what's visually in a chart or photo
rather than a text description of it.

## Setup

```bash
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-...
```

The first run downloads CLIP weights (~600MB for the default ViT-B-32
model), so it needs internet access once; after that it's fully local
except for the Claude API calls at generation time.

## Usage

```python
from multimodal_rag import MultimodalRAG

rag = MultimodalRAG()

# Ingest — extracts text chunks AND embedded images/figures from a PDF
rag.ingest_pdf("data/sample.pdf")

# Or index a folder of standalone images
rag.ingest_image_folder("data/images")

# Or a single image file
rag.ingest_image_file("data/images/diagram.png")

# Or plain text
rag.ingest_text_file("data/notes.txt")

# Query — retrieves relevant text + images, generates a grounded answer
result = rag.query("What trend does the Q3 revenue chart show?")
print(result["answer"])
print(result["sources"])   # which docs/pages/images were used
```

A Flask web UI is available from `app.py`:

```bash
pip install -r requirements.txt
python app.py
```

Then open `http://127.0.0.1:5000` in your browser to upload documents or images and ask questions.

See `example.py` for a runnable CLI demo.

## File structure

```
app.py
README.md
requirements.txt
example.py
.gitignore
src/
└── multimodal_rag/
    ├── config.py
    ├── document_processor.py
    ├── embeddings.py
    ├── vector_store.py
    ├── retriever.py
    ├── generator.py
    ├── rag_pipeline.py
    └── __init__.py
templates/
└── index.html
uploads/  # created at runtime
```
## Tuning knobs (`config.py`)

| Setting | Default | Notes |
|---|---|---|
| `clip_model_name` / `clip_pretrained` | `ViT-B-32` / `laion2b_s34b_b79k` | Swap to `ViT-L-14` for better quality if you have a GPU |
| `chunk_size` / `chunk_overlap` | 800 / 150 chars | Tune per document density |
| `top_k` | 5 | Retrieved text chunks (images retrieved at `top_k // 2`) |
| `anthropic_model` | `claude-sonnet-5` | Generation model |

## Extending this

- **Swap the vector store**: replace `vector_store.py` with a Pinecone/
  Weaviate/Qdrant client — the rest of the pipeline is agnostic to it.
- **Swap the embedder**: any model that can embed both text and images into
  one space works (e.g. SigLIP) — just implement `embed_text`/`embed_images`.
- **Add a reranker**: insert a cross-encoder reranking step in
  `retriever.py` after the initial CLIP retrieval for higher precision.
- **Audio/video**: extend `document_processor.py` with a transcription step
  (e.g. Whisper) that turns audio into indexable text chunks, and/or sample
  video frames as images.
