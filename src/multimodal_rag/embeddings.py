"""Unified text/image embeddings via CLIP.

Text and images are projected into the SAME vector space, so a text query
can directly retrieve relevant images (and an image query can retrieve
relevant text) without any translation step.
"""

from __future__ import annotations

import numpy as np
import torch
import open_clip
from PIL import Image

from .config import CONFIG


class ClipEmbedder:
    def __init__(self, config=CONFIG):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            config.clip_model_name, pretrained=config.clip_pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(config.clip_model_name)
        self.model.to(self.device).eval()

    @torch.no_grad()
    def embed_text(self, texts: list[str]) -> np.ndarray:
        tokens = self.tokenizer(texts).to(self.device)
        features = self.model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()

    @torch.no_grad()
    def embed_images(self, image_paths: list[str]) -> np.ndarray:
        batch = []
        for path in image_paths:
            with Image.open(path) as im:
                im = im.convert("RGB")
                batch.append(self.preprocess(im))
        tensor = torch.stack(batch).to(self.device)
        features = self.model.encode_image(tensor)
        features = features / features.norm(dim=-1, keepdim=True)
        return features.cpu().numpy()

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_text([text])[0]
