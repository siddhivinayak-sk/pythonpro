"""Tests for the LLM connection registry.

Uses a fake model factory so no provider SDK or network is required.
"""

from __future__ import annotations

from typing import Any

import pytest
from ai_agent_core.config.models import (
    Capability,
    ConnectionConfig,
    LLMConfig,
    ModelDescriptor,
    Provider,
)
from ai_agent_core.llm import (
    ConnectionNotFoundError,
    LLMConnectionRegistry,
    ModelNotFoundError,
)


class FakeFactory:
    """Records calls and returns a sentinel instead of a real model."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, dict[str, Any]]] = []

    def create_chat_model(
        self, connection: ConnectionConfig, model_name: str, **params: Any
    ) -> str:
        self.calls.append((connection.id, model_name, params))
        return f"model::{connection.id}::{model_name}"

    def create_embeddings(
        self, connection: ConnectionConfig, model_name: str, **params: Any
    ) -> str:
        self.calls.append((connection.id, model_name, params))
        return f"embed::{connection.id}::{model_name}"


def _config() -> LLMConfig:
    return LLMConfig(
        default_connection="ollama-local",
        connections=[
            ConnectionConfig(
                id="openai-prod",
                provider=Provider.OPENAI,
                api_key="sk-test",
                models=[
                    ModelDescriptor(
                        name="gpt-4o",
                        capabilities=[Capability.CHAT, Capability.VISION, Capability.TOOLS],
                    ),
                    ModelDescriptor(
                        name="text-embedding-3-large",
                        capabilities=[Capability.EMBEDDINGS],
                        dimensions=3072,
                    ),
                ],
            ),
            ConnectionConfig(
                id="ollama-local",
                provider=Provider.OLLAMA,
                base_url="http://ollama:11434",
                models=[
                    ModelDescriptor(
                        name="llama3.1", capabilities=[Capability.CHAT, Capability.TOOLS]
                    )
                ],
            ),
            ConnectionConfig(
                id="disabled-conn",
                provider=Provider.OPENAI,
                enabled=False,
                models=[ModelDescriptor(name="gpt-4o-mini")],
            ),
        ],
    )


def test_list_chat_models_across_connections_excludes_disabled_and_embeddings() -> None:
    reg = LLMConnectionRegistry.from_config(_config(), factory=FakeFactory())
    names = {(m.connection_id, m.model_name) for m in reg.list_chat_models()}
    assert ("openai-prod", "gpt-4o") in names
    assert ("ollama-local", "llama3.1") in names
    # embedding model excluded from chat listing; disabled connection excluded entirely
    assert ("openai-prod", "text-embedding-3-large") not in names
    assert all(cid != "disabled-conn" for cid, _ in names)


def test_list_embedding_models() -> None:
    reg = LLMConnectionRegistry.from_config(_config(), factory=FakeFactory())
    emb = reg.list_embedding_models()
    assert len(emb) == 1
    assert emb[0].model_name == "text-embedding-3-large"
    assert emb[0].dimensions == 3072


def test_default_connection_resolution() -> None:
    reg = LLMConnectionRegistry.from_config(_config(), factory=FakeFactory())
    assert reg.default_connection_id() == "ollama-local"


def test_default_connection_falls_back_to_first_enabled() -> None:
    cfg = _config()
    cfg.default_connection = None
    reg = LLMConnectionRegistry.from_config(cfg, factory=FakeFactory())
    assert reg.default_connection_id() == "openai-prod"


def test_get_chat_model_uses_factory_and_passes_params() -> None:
    factory = FakeFactory()
    reg = LLMConnectionRegistry.from_config(_config(), factory=factory)
    model = reg.get_chat_model("openai-prod", "gpt-4o", temperature=0.2)
    assert model == "model::openai-prod::gpt-4o"
    assert factory.calls == [("openai-prod", "gpt-4o", {"temperature": 0.2})]


def test_get_chat_model_defaults_connection_and_model() -> None:
    factory = FakeFactory()
    reg = LLMConnectionRegistry.from_config(_config(), factory=factory)
    model = reg.get_chat_model()  # default connection + its first chat model
    assert model == "model::ollama-local::llama3.1"


def test_get_chat_model_unknown_connection_raises() -> None:
    reg = LLMConnectionRegistry.from_config(_config(), factory=FakeFactory())
    with pytest.raises(ConnectionNotFoundError):
        reg.get_chat_model("does-not-exist", "gpt-4o")


def test_get_chat_model_no_chat_capable_model_raises() -> None:
    cfg = LLMConfig(
        connections=[
            ConnectionConfig(
                id="emb-only",
                provider=Provider.OPENAI,
                models=[ModelDescriptor(name="embed", capabilities=[Capability.EMBEDDINGS])],
            )
        ]
    )
    reg = LLMConnectionRegistry.from_config(cfg, factory=FakeFactory())
    with pytest.raises(ModelNotFoundError):
        reg.get_chat_model("emb-only")


def test_mark_unhealthy_excludes_connection() -> None:
    reg = LLMConnectionRegistry.from_config(_config(), factory=FakeFactory())
    reg.mark_unhealthy("openai-prod")
    assert reg.is_healthy("openai-prod") is False
    assert all(m.connection_id != "openai-prod" for m in reg.list_chat_models())
    reg.mark_healthy("openai-prod")
    assert any(m.connection_id == "openai-prod" for m in reg.list_chat_models())


def test_get_chat_model_without_any_connection_raises() -> None:
    reg = LLMConnectionRegistry.from_config(LLMConfig(), factory=FakeFactory())
    with pytest.raises(ConnectionNotFoundError):
        reg.get_chat_model()


def test_get_embeddings_explicit_connection_and_model() -> None:
    factory = FakeFactory()
    reg = LLMConnectionRegistry.from_config(_config(), factory=factory)
    model = reg.get_embeddings("openai-prod", "text-embedding-3-large")
    assert model == "embed::openai-prod::text-embedding-3-large"


def test_get_embeddings_finds_first_embeddings_model_across_connections() -> None:
    factory = FakeFactory()
    reg = LLMConnectionRegistry.from_config(_config(), factory=factory)
    # Default connection (ollama-local) has no embeddings model; resolution scans all connections.
    model = reg.get_embeddings()
    assert model == "embed::openai-prod::text-embedding-3-large"


def test_get_embeddings_no_embeddings_model_raises() -> None:
    cfg = LLMConfig(
        connections=[
            ConnectionConfig(
                id="chat-only",
                provider=Provider.OLLAMA,
                models=[ModelDescriptor(name="llama3.1", capabilities=[Capability.CHAT])],
            )
        ]
    )
    reg = LLMConnectionRegistry.from_config(cfg, factory=FakeFactory())
    with pytest.raises(ModelNotFoundError):
        reg.get_embeddings()
