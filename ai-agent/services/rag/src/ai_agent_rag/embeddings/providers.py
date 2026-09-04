"""Real (production) embedders. All SDK imports are lazy so the base package stays light.

- ``huggingface`` — sentence-transformers; supports Matryoshka truncation to the profile dimension.
- ``ollama`` — calls a local/remote Ollama server's embeddings endpoint over HTTP.
- ``openai`` / ``azure_openai`` / ``bedrock`` — via the shared ``ai_agent_core`` LangChain embeddings
  factory, so provider wiring (keys, Azure endpoint/version, Bedrock region/profile) lives in one place.
- ``postgresml`` — embeddings computed *inside* PostgreSQL via ``pgml.embed()`` (the PostgresML extension).

Every embedder applies the profile ``dimension``/``normalize`` post-processing so a collection's vectors
are consistent regardless of the backing model.
"""

from __future__ import annotations

from typing import Any

from ..config import EmbeddingProfile

Vector = list[float]

# Providers served by the shared LangChain embeddings factory in ai_agent_core.
_LANGCHAIN_PROVIDERS = frozenset({"openai", "azure_openai", "bedrock"})


def build_provider_embedder(profile: EmbeddingProfile, factory: Any = None):
    """Construct a non-hashing embedder for ``profile``. ``factory`` is injectable for tests."""
    if profile.provider == "huggingface":
        return HuggingFaceEmbedder(profile)
    if profile.provider == "ollama":
        return OllamaEmbedder(profile)
    if profile.provider in _LANGCHAIN_PROVIDERS:
        return LangChainEmbedder(profile, factory=factory)
    if profile.provider == "postgresml":
        return PostgresMLEmbedder(profile)
    raise ValueError(f"unknown embedding provider '{profile.provider}'")


def _truncate_normalize(vec: list[float], dimension: int, normalize: bool) -> list[float]:
    if len(vec) < dimension:
        raise ValueError(
            f"embedding produced {len(vec)} dimensions, fewer than the configured {dimension}; "
            "fix the profile 'dimension' to match the model or choose a larger model"
        )
    if len(vec) > dimension:  # Matryoshka-style truncation
        vec = vec[:dimension]
    if normalize:
        import math

        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
    return vec


def _normalize_dsn(dsn: str) -> str:
    """Accept SQLAlchemy-style DSNs and hand psycopg a plain ``postgresql://`` one."""
    return dsn.replace("postgresql+psycopg://", "postgresql://").replace(
        "postgresql+psycopg2://", "postgresql://"
    )


def _profile_to_connection(profile: EmbeddingProfile):
    """Map an embedding profile onto a core ``ConnectionConfig`` for the shared model factory."""
    from ai_agent_core.config.models import ConnectionConfig, Provider

    return ConnectionConfig(
        id=profile.id,
        provider=Provider(profile.provider),
        base_url=profile.base_url,
        api_key=profile.api_key,
        api_version=profile.api_version,
        region=profile.region,
        auth_profile=profile.auth_profile,
        extra=profile.extra or {},
    )


class HuggingFaceEmbedder:
    def __init__(self, profile: EmbeddingProfile) -> None:
        from sentence_transformers import SentenceTransformer  # lazy

        self.id = profile.id
        self.dimension = profile.dimension
        self.normalize = profile.normalize
        self.batch_size = profile.batch_size
        self._model = SentenceTransformer(profile.model, device=profile.device)

    def embed_documents(self, texts: list[str]) -> list[Vector]:
        vectors = self._model.encode(
            texts,
            normalize_embeddings=False,
            convert_to_numpy=False,
            batch_size=self.batch_size,
        )
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


class LangChainEmbedder:
    """OpenAI / Azure OpenAI / Bedrock embeddings via the shared ``ai_agent_core`` model factory.

    The factory is injectable so tests exercise the wrapping (delegation + truncate/normalize) with a fake
    embeddings model and no provider SDK.
    """

    def __init__(self, profile: EmbeddingProfile, factory: Any = None) -> None:
        self.id = profile.id
        self.dimension = profile.dimension
        self.normalize = profile.normalize
        if factory is None:
            from ai_agent_core.llm.factory import LangChainModelFactory

            factory = LangChainModelFactory()
        self._model = factory.create_embeddings(_profile_to_connection(profile), profile.model)

    def embed_documents(self, texts: list[str]) -> list[Vector]:
        raw = self._model.embed_documents(list(texts))
        return [
            _truncate_normalize([float(x) for x in v], self.dimension, self.normalize) for v in raw
        ]

    def embed_query(self, text: str) -> Vector:
        vec = [float(x) for x in self._model.embed_query(text)]
        return _truncate_normalize(vec, self.dimension, self.normalize)


class PostgresMLEmbedder:
    """Embeddings computed in-database by PostgresML's ``pgml.embed(transformer, text)``.

    ``connector`` returns a psycopg-style connection (context manager with ``.execute() -> cursor``); it is
    injectable so tests can run without a live PostgresML instance.
    """

    def __init__(self, profile: EmbeddingProfile, connector: Any = None) -> None:
        self.id = profile.id
        self.dimension = profile.dimension
        self.normalize = profile.normalize
        self.model = profile.model
        self._dsn = _normalize_dsn(profile.dsn or "")
        if not self._dsn:
            raise ValueError("postgresml embedder requires a dsn (embedding profile 'dsn')")
        self._connector = connector or self._default_connector

    def _default_connector(self):
        import psycopg

        return psycopg.connect(self._dsn, autocommit=True)

    def _embed(self, text: str) -> Vector:
        with self._connector() as conn:
            cur = conn.execute("SELECT pgml.embed(%s, %s)", (self.model, text))
            row = cur.fetchone()
        if not row or row[0] is None:
            raise RuntimeError("pgml.embed returned no vector")
        vec = [float(x) for x in row[0]]
        return _truncate_normalize(vec, self.dimension, self.normalize)

    def embed_documents(self, texts: list[str]) -> list[Vector]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> Vector:
        return self._embed(text)
