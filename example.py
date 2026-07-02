"""
Example usage of the multimodal RAG system.

Setup:
    pip install -r requirements.txt
    export ANTHROPIC_API_KEY=sk-...

Run:
    python example.py
"""

import sys
sys.path.insert(0, "src")

from multimodal_rag import MultimodalRAG


def main():
    rag = MultimodalRAG()

    # --- Ingest ---
    # A PDF: indexes text chunks AND embedded images/figures.
    n = rag.ingest_pdf("data/sample.pdf")
    print(f"Indexed {n} chunks/images from sample.pdf")

    # A folder of standalone images (e.g. product photos, diagrams).
    # n = rag.ingest_image_folder("data/images")

    print("Total indexed items:", rag.stats()["total_indexed"])

    # --- Query ---
    result = rag.query("What does the chart on page 2 show, and how does it relate to the text?")
    print("\nANSWER:\n", result["answer"])
    print("\nSOURCES:\n", result["sources"])


if __name__ == "__main__":
    main()
