"""Tests for the embeddings abstraction + hashing embedder."""

from __future__ import annotations

import math

import pytest
from ai_agent_rag.config import EmbeddingProfile
from ai_agent_rag.embeddings import HashingEmbedder, build_embedder


def _cos(a, b):
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def test_hashing_dimension_and_determinism() -> None:
    emb = HashingEmbedder("e", dimension=64)
    v1 = emb.embed_query("the quick brown fox")
    v2 = emb.embed_query("the quick brown fox")
    assert len(v1) == 64
    assert v1 == v2  # deterministic


def test_hashing_similarity_reflects_shared_tokens() -> None:
    emb = HashingEmbedder("e", dimension=512)
    q = emb.embed_query("parental leave policy for employees")
    close = emb.embed_query("what is the parental leave policy")
    far = emb.embed_query("kubernetes networking ingress controller")
    assert _cos(q, close) > _cos(q, far)


def test_build_embedder_hashing() -> None:
    emb = build_embedder(EmbeddingProfile(id="e", provider="hashing", dimension=128))
    assert emb.dimension == 128
    assert isinstance(emb, HashingEmbedder)


def test_build_embedder_unknown_provider() -> None:
    with pytest.raises(ValueError):
        build_embedder(EmbeddingProfile(id="e", provider="bogus"))
