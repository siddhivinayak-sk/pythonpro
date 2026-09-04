"""A deterministic, dependency-free embedder for tests and offline development.

It hashes tokens into a fixed-dimension vector (signed hashing trick) and L2-normalises. Texts that share
tokens get higher cosine similarity, which is enough to exercise the full retrieval path without
downloading a model. Not for production quality — use a real profile (huggingface/ollama) there.
"""

from __future__ import annotations

import hashlib
import math
import re

_TOKEN = re.compile(r"[a-z0-9]+")


class HashingEmbedder:
    def __init__(self, id: str, dimension: int = 256, normalize: bool = True) -> None:
        self.id = id
        self.dimension = dimension
        self.normalize = normalize

    def _embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dimension
        for token in _TOKEN.findall(text.lower()):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            h = int.from_bytes(digest, "big")
            index = h % self.dimension
            sign = 1.0 if (h >> 17) & 1 else -1.0
            vec[index] += sign
        if self.normalize:
            norm = math.sqrt(sum(v * v for v in vec))
            if norm > 0:
                vec = [v / norm for v in vec]
        return vec

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)
