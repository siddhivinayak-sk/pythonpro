"""RAG configuration.

Two layers:
- ``RagSettings`` — service-level settings from env (``RAG_*``): host/port, the path to the pipeline YAML,
  the index-metadata DB location, and a fallback vector-store backend for out-of-the-box runs.
- ``RagConfig`` — the full pipeline definition (embeddings, chunkers, sources, collections, indexer),
  normally loaded from a YAML file (see ``config/rag.example.yaml`` and docs/subprojects/rag.md §10).

When no YAML is supplied the app synthesises a minimal, runnable config from ``RagSettings`` (default:
in-memory store + hashing embedder) so the service boots with no external dependencies.
"""

from __future__ import annotations

from ai_agent_core.config import BaseServiceSettings
from pydantic import BaseModel, Field
from pydantic_settings import SettingsConfigDict


# --- pipeline config (YAML) --------------------------------------------------
class EmbeddingProfile(BaseModel):
    id: str
    provider: str = "hashing"  # hashing | huggingface | ollama | openai
    model: str = "hashing-256"
    dimension: int = 256
    normalize: bool = True
    device: str = "cpu"
    batch_size: int = 32
    base_url: str | None = None  # ollama / openai-compatible


class ChunkerProfile(BaseModel):
    id: str
    strategy: str = "recursive"  # fixed | recursive | sentence | markdown
    target_tokens: int = 512
    overlap_tokens: int = 0
    max_tokens: int = 1024


class EmbeddingsConfig(BaseModel):
    profiles: list[EmbeddingProfile] = Field(default_factory=list)


class ChunkersConfig(BaseModel):
    profiles: list[ChunkerProfile] = Field(default_factory=list)


class DirectorySource(BaseModel):
    id: str
    type: str = "directory"
    path: str
    recursive: bool = True
    include: list[str] = Field(default_factory=lambda: ["**/*"])
    exclude: list[str] = Field(default_factory=list)


class CollectionSpec(BaseModel):
    name: str
    embedding_ref: str
    dimension: int
    distance: str = "cosine"  # cosine | l2 | ip
    chunker_ref: str
    hybrid: bool = False
    sources: list[str] = Field(default_factory=list)


class IndexJob(BaseModel):
    collection: str
    cron: str = "0 2 * * *"
    mode: str = "incremental"  # incremental | full


class IndexerConfig(BaseModel):
    timezone: str = "UTC"
    jobs: list[IndexJob] = Field(default_factory=list)
    on_start_run: bool = False


class PgVectorConfig(BaseModel):
    dsn: str | None = None


class ChromaConfig(BaseModel):
    path: str | None = None


class VectorStoreConfig(BaseModel):
    backend: str = "pgvector"  # memory | chroma | pgvector | qdrant | milvus
    pgvector: PgVectorConfig = Field(default_factory=PgVectorConfig)
    chroma: ChromaConfig = Field(default_factory=ChromaConfig)


class RetrievalConfig(BaseModel):
    default_k: int = 6
    rerank_enabled: bool = False


class RagConfig(BaseModel):
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    chunkers: ChunkersConfig = Field(default_factory=ChunkersConfig)
    sources: list[DirectorySource] = Field(default_factory=list)
    collections: list[CollectionSpec] = Field(default_factory=list)
    indexer: IndexerConfig = Field(default_factory=IndexerConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)

    # -- resolution helpers --
    def embedding(self, ref: str) -> EmbeddingProfile:
        for profile in self.embeddings.profiles:
            if profile.id == ref:
                return profile
        raise KeyError(f"embedding profile '{ref}' not found")

    def chunker(self, ref: str) -> ChunkerProfile:
        for profile in self.chunkers.profiles:
            if profile.id == ref:
                return profile
        raise KeyError(f"chunker profile '{ref}' not found")

    def collection(self, name: str) -> CollectionSpec:
        for spec in self.collections:
            if spec.name == name:
                return spec
        raise KeyError(f"collection '{name}' not found")

    def sources_for(self, collection: str) -> list[DirectorySource]:
        spec = self.collection(collection)
        by_id = {s.id: s for s in self.sources}
        return [by_id[sid] for sid in spec.sources if sid in by_id]


# --- service settings (env) --------------------------------------------------
class RagSettings(BaseServiceSettings):
    model_config = SettingsConfigDict(env_prefix="RAG_", env_nested_delimiter="__", extra="ignore")

    service_name: str = "rag-api"
    host: str = "0.0.0.0"
    port: int = 8081

    # Path to the full pipeline YAML. If unset, a minimal in-memory config is synthesised.
    config_file: str | None = None

    # Index-metadata DB (change detection + run history).
    index_db_path: str = ":memory:"

    # Fallback vector store used only when no config_file is provided.
    vector_store_backend: str = "memory"
    pgvector_dsn: str | None = None

    # Hardening: per-client requests/minute (0 disables). Health + /metrics are exempt.
    rate_limit_per_minute: int = 0


def default_rag_config(settings: RagSettings) -> RagConfig:
    """A minimal, runnable config used when no YAML is supplied.

    Uses a hashing embedder (no model download) and the configured fallback backend, with a single
    ``default`` collection so the service is immediately usable for local dev and tests.
    """
    return RagConfig(
        vector_store=VectorStoreConfig(
            backend=settings.vector_store_backend,
            pgvector=PgVectorConfig(dsn=settings.pgvector_dsn),
        ),
        embeddings=EmbeddingsConfig(
            profiles=[EmbeddingProfile(id="default-embed", provider="hashing", dimension=256)]
        ),
        chunkers=ChunkersConfig(
            profiles=[ChunkerProfile(id="default-chunk", strategy="recursive", target_tokens=256)]
        ),
        collections=[
            CollectionSpec(
                name="default",
                embedding_ref="default-embed",
                dimension=256,
                chunker_ref="default-chunk",
            )
        ],
    )
