"""The LLM connection registry.

Loads connections from config, enumerates the models they expose (across *all* enabled connections), and
hands out ready-to-use chat models on request. This is the heart of the multi-provider, multi-connection,
runtime-selection requirement (design doc §6).
"""

from __future__ import annotations

from typing import Any

from ..config.models import Capability, ConnectionConfig, LLMConfig
from .factory import LangChainModelFactory, ModelFactory
from .types import ResolvedModel


class ConnectionNotFoundError(KeyError):
    """Raised when a connection id is not registered."""


class ModelNotFoundError(KeyError):
    """Raised when a connection exposes no suitable model."""


class LLMConnectionRegistry:
    """Registry over the configured LLM connections."""

    def __init__(self, config: LLMConfig, factory: ModelFactory | None = None) -> None:
        self._config = config
        self._factory: ModelFactory = factory or LangChainModelFactory()
        self._connections: dict[str, ConnectionConfig] = {c.id: c for c in config.connections}
        self._unhealthy: set[str] = set()

    @classmethod
    def from_config(
        cls, config: LLMConfig, factory: ModelFactory | None = None
    ) -> LLMConnectionRegistry:
        return cls(config, factory)

    # -- introspection -----------------------------------------------------
    def list_connections(self, *, enabled_only: bool = True) -> list[ConnectionConfig]:
        conns = list(self._connections.values())
        if enabled_only:
            conns = [c for c in conns if c.enabled and c.id not in self._unhealthy]
        return conns

    def _resolve_models(self, connection: ConnectionConfig) -> list[ResolvedModel]:
        return [
            ResolvedModel(
                connection_id=connection.id,
                provider=connection.provider,
                model_name=m.name,
                display_name=m.label,
                capabilities=tuple(m.capabilities),
                dimensions=m.dimensions,
                context_window=m.context_window,
            )
            for m in connection.models
        ]

    def list_models(
        self,
        *,
        capability: Capability | None = Capability.CHAT,
        enabled_only: bool = True,
    ) -> list[ResolvedModel]:
        """List models across all connections, optionally filtered by capability."""
        result: list[ResolvedModel] = []
        for connection in self.list_connections(enabled_only=enabled_only):
            for resolved in self._resolve_models(connection):
                if capability is None or resolved.has(capability):
                    result.append(resolved)
        return result

    def list_chat_models(self) -> list[ResolvedModel]:
        return self.list_models(capability=Capability.CHAT)

    def list_embedding_models(self) -> list[ResolvedModel]:
        return self.list_models(capability=Capability.EMBEDDINGS)

    def get_connection(self, connection_id: str) -> ConnectionConfig:
        try:
            return self._connections[connection_id]
        except KeyError:
            raise ConnectionNotFoundError(connection_id) from None

    def default_connection_id(self) -> str | None:
        if self._config.default_connection:
            return self._config.default_connection
        enabled = self.list_connections()
        return enabled[0].id if enabled else None

    # -- health ------------------------------------------------------------
    def mark_unhealthy(self, connection_id: str) -> None:
        """Exclude a connection from listings/resolution without removing its config."""
        self._unhealthy.add(connection_id)

    def mark_healthy(self, connection_id: str) -> None:
        self._unhealthy.discard(connection_id)

    def is_healthy(self, connection_id: str) -> bool:
        return connection_id not in self._unhealthy

    # -- resolution --------------------------------------------------------
    def get_chat_model(
        self,
        connection_id: str | None = None,
        model_name: str | None = None,
        **params: Any,
    ) -> Any:
        """Return a ready chat model for the given connection/model.

        Falls back to the default connection and its first chat-capable model when not specified.
        ``params`` (temperature, max_tokens, ...) pass through to the underlying model.
        """
        connection_id = connection_id or self.default_connection_id()
        if connection_id is None:
            raise ConnectionNotFoundError("no LLM connections are configured")

        connection = self.get_connection(connection_id)

        if model_name is None:
            chat_models = [m for m in self._resolve_models(connection) if m.has(Capability.CHAT)]
            if not chat_models:
                raise ModelNotFoundError(
                    f"connection '{connection_id}' exposes no chat-capable model; specify model_name"
                )
            model_name = chat_models[0].model_name

        return self._factory.create_chat_model(connection, model_name, **params)

    def get_embeddings(
        self,
        connection_id: str | None = None,
        model_name: str | None = None,
        **params: Any,
    ) -> Any:
        """Return a ready embeddings model.

        When neither is specified, the first embeddings-capable model across all enabled connections is
        used. Raises ``ModelNotFoundError`` when no embeddings model is configured.
        """
        if connection_id is None and model_name is None:
            for conn in self.list_connections():
                embed_models = [
                    m for m in self._resolve_models(conn) if m.has(Capability.EMBEDDINGS)
                ]
                if embed_models:
                    return self._factory.create_embeddings(
                        conn, embed_models[0].model_name, **params
                    )
            raise ModelNotFoundError("no embeddings-capable model is configured")

        connection_id = connection_id or self.default_connection_id()
        if connection_id is None:
            raise ConnectionNotFoundError("no LLM connections are configured")
        connection = self.get_connection(connection_id)

        if model_name is None:
            embed_models = [
                m for m in self._resolve_models(connection) if m.has(Capability.EMBEDDINGS)
            ]
            if not embed_models:
                raise ModelNotFoundError(
                    f"connection '{connection_id}' exposes no embeddings model; specify model_name"
                )
            model_name = embed_models[0].model_name

        return self._factory.create_embeddings(connection, model_name, **params)
