"""Central configuration for the multimodal RAG system."""

import os
from dataclasses import dataclass


@dataclass
class Config:
    # --- Embedding model (CLIP) ---
    # ViT-B-32 / laion2b_s34b_b79k gives a good speed/quality tradeoff and
    # runs fine on CPU. Swap to "ViT-L-14" + "laion2b_s32b_b82k" for higher
    # quality if you have a GPU.
    clip_model_name: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"

    # --- Chunking ---
    chunk_size: int = 800        # characters per text chunk
    chunk_overlap: int = 150     # character overlap between chunks

    # --- Storage ---
    persist_dir: str = "./chroma_db"
    image_store_dir: str = "./data/images"
    collection_name: str = "multimodal_rag"

    # --- Retrieval ---
    top_k: int = 5

    # --- Generation ---
    # Model string for the Anthropic Messages API.
    anthropic_model: str = "claude-sonnet-5"
    max_tokens: int = 1024
    anthropic_api_key_env: str = "ANTHROPIC_API_KEY"


CONFIG = Config()

os.makedirs(CONFIG.persist_dir, exist_ok=True)
os.makedirs(CONFIG.image_store_dir, exist_ok=True)
