"""Typed configuration models shared across the platform.

These model the *LLM connection registry* configuration described in the design doc
(docs/comprehensive-analysis.md §6): multiple named connections per provider, each exposing models with
capability flags.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class Provider(StrEnum):
    """A kind of LLM backend."""

    OPENAI = "openai"
    AZURE_OPENAI = "azure_openai"
    BEDROCK = "bedrock"
    OLLAMA = "ollama"


class Capability(StrEnum):
    """What a model can do. Drives UX (e.g. vision -> allow image upload)."""

    CHAT = "chat"
    EMBEDDINGS = "embeddings"
    VISION = "vision"
    TOOLS = "tools"


class ModelDescriptor(BaseModel):
    """A model exposed by a connection."""

    name: str
    display_name: str | None = None
    capabilities: list[Capability] = Field(default_factory=lambda: [Capability.CHAT])
    dimensions: int | None = None  # embedding output dimension (embedding models)
    context_window: int | None = None

    def has(self, capability: Capability) -> bool:
        return capability in self.capabilities

    @property
    def label(self) -> str:
        return self.display_name or self.name


class ConnectionConfig(BaseModel):
    """A named, configured instance of a provider.

    Multiple connections may target the same provider (e.g. ``openai-prod`` and ``openai-sandbox``).
    Provider-specific fields that do not apply are simply ignored by the factory.
    """

    id: str
    provider: Provider
    enabled: bool = True

    # Common optional connection fields.
    base_url: str | None = (
        None  # OpenAI-compatible gateways, Ollama server, or Azure OpenAI endpoint
    )
    api_key: str | None = None  # OpenAI / Azure OpenAI (supports ${ENV} expansion via load_config)
    region: str | None = None  # Bedrock region
    auth_profile: str | None = None  # Bedrock/AWS named profile
    api_version: str | None = None  # Azure OpenAI REST API version (e.g. "2024-10-21")
    extra: dict[str, Any] = Field(default_factory=dict)  # passthrough kwargs to the model factory

    # Declared/allowlisted models. Live discovery (Ollama /api/tags, etc.) is a later enhancement.
    models: list[ModelDescriptor] = Field(default_factory=list)


class LLMConfig(BaseModel):
    """Top-level LLM configuration: the set of connections and an optional default."""

    default_connection: str | None = None
    connections: list[ConnectionConfig] = Field(default_factory=list)
