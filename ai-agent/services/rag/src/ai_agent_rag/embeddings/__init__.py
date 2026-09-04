"""Embedding models (configurable provider + dimension)."""

from __future__ import annotations

from .base import Embedder, Vector, build_embedder
from .hashing import HashingEmbedder

__all__ = ["Embedder", "Vector", "build_embedder", "HashingEmbedder"]
