"""Extracts text chunks and images from source documents.

Supports:
  - PDFs (text per page, chunked; embedded images saved to disk)
  - Folders of standalone images
  - Plain .txt files
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from typing import Any, Literal

import fitz  # PyMuPDF
from PIL import Image

from .config import CONFIG


@dataclass
class Document:
    """A single retrievable unit: either a text chunk or an image."""

    id: str
    type: Literal["text", "image"]
    content: str            # text content, OR filesystem path for images
    metadata: dict[str, Any] = field(default_factory=dict)


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Simple sliding-window character chunker with sentence-ish boundaries."""
    text = " ".join(text.split())  # normalize whitespace
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        # try to break on a sentence/space boundary near the end
        if end < len(text):
            boundary = text.rfind(". ", start, end)
            if boundary == -1 or boundary < start + chunk_size * 0.5:
                boundary = text.rfind(" ", start, end)
            if boundary != -1 and boundary > start:
                end = boundary + 1
        chunks.append(text[start:end].strip())
        start = end - overlap if end - overlap > start else end
    return [c for c in chunks if c]


class DocumentProcessor:
    def __init__(self, config=CONFIG):
        self.config = config

    def process_pdf(self, pdf_path: str) -> list[Document]:
        """Extract text chunks and embedded images from every page of a PDF."""
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(pdf_path)

        docs: list[Document] = []
        source_name = os.path.basename(pdf_path)
        pdf = fitz.open(pdf_path)

        try:
            for page_index in range(len(pdf)):
                page = pdf[page_index]

                # --- text ---
                page_text = page.get_text()
                for chunk in chunk_text(page_text, self.config.chunk_size, self.config.chunk_overlap):
                    docs.append(
                        Document(
                            id=str(uuid.uuid4()),
                            type="text",
                            content=chunk,
                            metadata={
                                "source": source_name,
                                "page": page_index + 1,
                            },
                        )
                    )

                # --- images ---
                for img_index, img_info in enumerate(page.get_images(full=True)):
                    xref = img_info[0]
                    try:
                        base_image = pdf.extract_image(xref)
                    except Exception:
                        continue
                    img_bytes = base_image["image"]
                    ext = base_image.get("ext", "png")

                    # Skip tiny images (icons, bullets, decorations) — not
                    # useful retrieval targets and add noise.
                    if len(img_bytes) < 5_000:
                        continue

                    fname = f"{uuid.uuid4()}.{ext}"
                    fpath = os.path.join(self.config.image_store_dir, fname)
                    with open(fpath, "wb") as f:
                        f.write(img_bytes)

                    docs.append(
                        Document(
                            id=str(uuid.uuid4()),
                            type="image",
                            content=fpath,
                            metadata={
                                "source": source_name,
                                "page": page_index + 1,
                                "image_index": img_index,
                            },
                        )
                    )
        finally:
            pdf.close()

        return docs

    def process_image_folder(self, folder_path: str) -> list[Document]:
        """Index every standalone image in a folder."""
        docs: list[Document] = []
        valid_ext = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        for fname in sorted(os.listdir(folder_path)):
            ext = os.path.splitext(fname)[1].lower()
            if ext not in valid_ext:
                continue
            fpath = os.path.join(folder_path, fname)
            try:
                with Image.open(fpath) as im:
                    im.verify()
            except Exception:
                continue
            docs.append(
                Document(
                    id=str(uuid.uuid4()),
                    type="image",
                    content=fpath,
                    metadata={"source": fname},
                )
            )
        return docs

    def process_image_file(self, image_path: str) -> list[Document]:
        """Index a single image file."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(image_path)

        valid_ext = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
        ext = os.path.splitext(image_path)[1].lower()
        if ext not in valid_ext:
            raise ValueError(f"Unsupported image format: {ext}")

        try:
            with Image.open(image_path) as im:
                im.verify()
        except Exception as exc:
            raise RuntimeError(f"Cannot read image file: {image_path}") from exc

        return [
            Document(
                id=str(uuid.uuid4()),
                type="image",
                content=image_path,
                metadata={"source": os.path.basename(image_path)},
            )
        ]

    def process_text_file(self, txt_path: str) -> list[Document]:
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        source_name = os.path.basename(txt_path)
        return [
            Document(
                id=str(uuid.uuid4()),
                type="text",
                content=chunk,
                metadata={"source": source_name},
            )
            for chunk in chunk_text(text, self.config.chunk_size, self.config.chunk_overlap)
        ]
