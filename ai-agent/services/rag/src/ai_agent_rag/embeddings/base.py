"""Embedding abstraction: a provider-agnostic ``Embedder`` plus a factory.

Indexing and querying must use the *same* embedder for a collection, so the registry resolves an embedder
from a named profile and the collection records which profile it was built with.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..config import EmbeddingProfile

Vector = list[float]


@runtime_checkable
class Embedder(Protocol):
    id: str
    dimension: int

    def embed_documents(self, texts: list[str]) -> list[Vector]: ...

    def embed_query(self, text: str) -> Vector: ...


def build_embedder(profile: EmbeddingProfile) -> Embedder:
    """Construct an embedder from a profile. Provider SDKs are imported lazily."""
    if profile.provider == "hashing":
        from .hashing import HashingEmbedder

        return HashingEmbedder(profile.id, dimension=profile.dimension, normalize=profile.normalize)

    if profile.provider in ("huggingface", "ollama", "openai"):
        from .providers import build_provider_embedder

        return build_provider_embedder(profile)

    raise ValueError(f"unknown embedding provider '{profile.provider}'")
