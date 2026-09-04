"""Real (production) embedders. All SDK imports are lazy so the base package stays light.

- ``huggingface`` — sentence-transformers; supports Matryoshka truncation to the profile dimension.
- ``ollama`` — calls a local/remote Ollama server's embeddings endpoint over HTTP.
- ``openai`` — reserved; raises with guidance until wired in a later pass.
"""

from __future__ import annotations

from ..config import EmbeddingProfile

Vector = list[float]


def build_provider_embedder(profile: EmbeddingProfile):
    if profile.provider == "huggingface":
        return HuggingFaceEmbedder(profile)
    if profile.provider == "ollama":
        return OllamaEmbedder(profile)
    if profile.provider == "openai":
        raise NotImplementedError(
            "openai embeddings are reserved for a later pass; use huggingface or ollama"
        )
    raise ValueError(f"unknown provider '{profile.provider}'")


def _truncate_normalize(vec: list[float], dimension: int, normalize: bool) -> list[float]:
    if len(vec) > dimension:  # Matryoshka-style truncation
        vec = vec[:dimension]
    if normalize:
        import math

        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
    return vec


class HuggingFaceEmbedder:
    def __init__(self, profile: EmbeddingProfile) -> None:
        from sentence_transformers import SentenceTransformer  # lazy

        self.id = profile.id
        self.dimension = profile.dimension
        self.normalize = profile.normalize
        self._model = SentenceTransformer(profile.model, device=profile.device)

    def embed_documents(self, texts: list[str]) -> list[Vector]:
        vectors = self._model.encode(texts, normalize_embeddings=False, convert_to_numpy=False)
        return [
            _truncate_normalize(list(map(float, v)), self.dimension, self.normalize)
            for v in vectors
        ]

    def embed_query(self, text: str) -> Vector:
        return self.embed_documents([text])[0]


class OllamaEmbedder:
    def __init__(self, profile: EmbeddingProfile) -> None:
        self.id = profile.id
        self.dimension = profile.dimension
        self.normalize = profile.normalize
        self.model = profile.model
        self.base_url = (profile.base_url or "http://localhost:11434").rstrip("/")

    def _embed(self, text: str) -> Vector:
        import httpx

        resp = httpx.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=60,
        )
        resp.raise_for_status()
        vec = [float(x) for x in resp.json()["embedding"]]
        return _truncate_normalize(vec, self.dimension, self.normalize)

    def embed_documents(self, texts: list[str]) -> list[Vector]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> Vector:
        return self._embed(text)
