"""Generates a grounded answer from retrieved text + image context.

Retrieved images are sent to Claude as actual image content blocks (not
just captions/filenames), so the model can reason over what's visually in
them — charts, diagrams, photos, etc.
"""

from __future__ import annotations

import base64
import mimetypes
import os

import anthropic

from .config import CONFIG


SYSTEM_PROMPT = """You are a precise research assistant. Answer the user's \
question using ONLY the provided context (text excerpts and images). If the \
context does not contain enough information to answer, say so explicitly \
rather than guessing. When you use a piece of context, cite its source and \
page number in parentheses, e.g. (source.pdf, p.3)."""


class AnswerGenerator:
    def __init__(self, config=CONFIG):
        self.config = config
        api_key = os.environ.get(config.anthropic_api_key_env)
        if not api_key:
            raise RuntimeError(
                f"Set the {config.anthropic_api_key_env} environment variable "
                "with your Anthropic API key before generating answers."
            )
        self.client = anthropic.Anthropic(api_key=api_key)

    @staticmethod
    def _encode_image(path: str) -> dict:
        mime, _ = mimetypes.guess_type(path)
        mime = mime or "image/png"
        with open(path, "rb") as f:
            data = base64.standard_b64encode(f.read()).decode("utf-8")
        return {
            "type": "image",
            "source": {"type": "base64", "media_type": mime, "data": data},
        }

    def _build_context_blocks(self, retrieved: dict) -> list[dict]:
        blocks: list[dict] = []

        if retrieved["text"]:
            text_context = "\n\n".join(
                f"[{hit['metadata'].get('source', 'unknown')}"
                f"{', p.' + str(hit['metadata']['page']) if 'page' in hit['metadata'] else ''}]\n"
                f"{hit['content']}"
                for hit in retrieved["text"]
            )
            blocks.append({"type": "text", "text": f"TEXT CONTEXT:\n{text_context}"})

        for hit in retrieved["images"]:
            path = hit["content"]
            if not os.path.exists(path):
                continue
            label = (
                f"IMAGE CONTEXT [{hit['metadata'].get('source', 'unknown')}"
                f"{', p.' + str(hit['metadata']['page']) if 'page' in hit['metadata'] else ''}]:"
            )
            blocks.append({"type": "text", "text": label})
            blocks.append(self._encode_image(path))

        return blocks

    def generate(self, query: str, retrieved: dict) -> str:
        context_blocks = self._build_context_blocks(retrieved)
        if not context_blocks:
            return "No relevant context was found for this query."

        message_content = context_blocks + [
            {"type": "text", "text": f"\nQUESTION: {query}"}
        ]

        response = self.client.messages.create(
            model=self.config.anthropic_model,
            max_tokens=self.config.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": message_content}],
        )
        return "".join(block.text for block in response.content if block.type == "text")
