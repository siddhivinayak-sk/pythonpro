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


# --- provider embedders (openai / azure_openai / bedrock / postgresml), offline via fakes -----------
from ai_agent_core.config.models import Provider  # noqa: E402
from ai_agent_rag.embeddings.providers import (  # noqa: E402
    LangChainEmbedder,
    PostgresMLEmbedder,
    _profile_to_connection,
    build_provider_embedder,
)


class _FakeLCEmbeddings:
    """Stand-in for a LangChain embeddings model (returns a length-based vector)."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [[float(len(t)), 2.0, 0.0] for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 2.0, 0.0]


class _FakeFactory:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def create_embeddings(self, connection, model_name: str, **params):
        self.calls.append((connection.provider, connection.id, model_name))
        return _FakeLCEmbeddings()


def test_profile_to_connection_azure() -> None:
    conn = _profile_to_connection(
        EmbeddingProfile(
            id="e",
            provider="azure_openai",
            model="text-embedding-3-large",
            base_url="https://r.openai.azure.com/",
            api_key="k",
            api_version="2024-10-21",
        )
    )
    assert conn.provider == Provider.AZURE_OPENAI
    assert conn.base_url == "https://r.openai.azure.com/"
    assert conn.api_key == "k"
    assert conn.api_version == "2024-10-21"


def test_profile_to_connection_bedrock() -> None:
    conn = _profile_to_connection(
        EmbeddingProfile(
            id="e",
            provider="bedrock",
            model="amazon.titan-embed-text-v2:0",
            region="us-east-1",
            auth_profile="default",
        )
    )
    assert conn.provider == Provider.BEDROCK
    assert conn.region == "us-east-1"
    assert conn.auth_profile == "default"


def test_langchain_embedder_truncates() -> None:
    profile = EmbeddingProfile(
        id="e", provider="openai", model="m", dimension=2, normalize=False, api_key="k"
    )
    factory = _FakeFactory()
    emb = build_provider_embedder(profile, factory=factory)
    assert isinstance(emb, LangChainEmbedder)
    assert emb.embed_query("abcd") == [4.0, 2.0]  # len 4 -> [4,2,0] truncated to dim 2
    assert factory.calls[0][2] == "m"


def test_langchain_embedder_normalizes() -> None:
    profile = EmbeddingProfile(
        id="e", provider="openai", model="m", dimension=3, normalize=True, api_key="k"
    )
    emb = build_provider_embedder(profile, factory=_FakeFactory())
    n = math.sqrt(2.0**2 + 2.0**2)
    assert emb.embed_query("ab") == [2.0 / n, 2.0 / n, 0.0]


def test_postgresml_embedder_with_fake_connector() -> None:
    class _FakeCursor:
        def fetchone(self):
            return ([1.0, 2.0, 3.0],)

    class _FakeConn:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def execute(self, sql, params):
            return _FakeCursor()

    profile = EmbeddingProfile(
        id="e",
        provider="postgresml",
        model="intfloat/e5-small",
        dimension=2,
        normalize=False,
        dsn="postgresql+psycopg://u@h/db",
    )
    emb = PostgresMLEmbedder(profile, connector=lambda: _FakeConn())
    assert emb.embed_query("x") == [1.0, 2.0]  # truncated to dim 2


def test_postgresml_requires_dsn() -> None:
    with pytest.raises(ValueError):
        PostgresMLEmbedder(EmbeddingProfile(id="e", provider="postgresml", model="m"))


def test_build_embedder_routes_postgresml() -> None:
    emb = build_embedder(
        EmbeddingProfile(id="e", provider="postgresml", model="m", dsn="postgresql://u@h/db")
    )
    assert isinstance(emb, PostgresMLEmbedder)


def test_truncate_normalize_raises_on_short_vector() -> None:
    from ai_agent_rag.embeddings.providers import _truncate_normalize

    with pytest.raises(ValueError, match="fewer than the configured"):
        _truncate_normalize([1.0, 2.0], dimension=4, normalize=False)


def test_truncate_normalize_truncates_long_vector() -> None:
    from ai_agent_rag.embeddings.providers import _truncate_normalize

    assert _truncate_normalize([1.0, 2.0, 3.0, 4.0], dimension=2, normalize=False) == [1.0, 2.0]
