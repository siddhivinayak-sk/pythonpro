"""Model factory: turns a connection + model name into a ready LangChain chat model.

LangChain is imported lazily so that importing this module (and the registry) does not require any
provider SDK. Only :meth:`LangChainModelFactory.create_chat_model` needs LangChain installed.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from ..config.models import ConnectionConfig, Provider

# Maps our Provider enum to the ``model_provider`` string understood by ``init_chat_model``.
_PROVIDER_TO_LANGCHAIN: dict[Provider, str] = {
    Provider.OPENAI: "openai",
    Provider.AZURE_OPENAI: "azure_openai",
    Provider.BEDROCK: "bedrock_converse",
    Provider.OLLAMA: "ollama",
}

# Embeddings use different ``provider`` strings than chat (e.g. Bedrock is ``bedrock``, not
# ``bedrock_converse``) per ``langchain.embeddings.init_embeddings``.
_PROVIDER_TO_LANGCHAIN_EMBEDDINGS: dict[Provider, str] = {
    Provider.OPENAI: "openai",
    Provider.AZURE_OPENAI: "azure_openai",
    Provider.BEDROCK: "bedrock",
    Provider.OLLAMA: "ollama",
}


@runtime_checkable
class ModelFactory(Protocol):
    """Creates chat / embedding model instances for a connection. Injectable for testing."""

    def create_chat_model(
        self, connection: ConnectionConfig, model_name: str, **params: Any
    ) -> Any: ...

    def create_embeddings(
        self, connection: ConnectionConfig, model_name: str, **params: Any
    ) -> Any: ...


class LangChainModelFactory:
    """Default factory using LangChain's provider-agnostic ``init_chat_model``."""

    def create_chat_model(
        self, connection: ConnectionConfig, model_name: str, **params: Any
    ) -> Any:
        from langchain.chat_models import init_chat_model  # lazy import

        provider, kwargs = self._chat_args(connection, model_name, **params)
        return init_chat_model(model_name, model_provider=provider, **kwargs)

    def create_embeddings(
        self, connection: ConnectionConfig, model_name: str, **params: Any
    ) -> Any:
        from langchain.embeddings import init_embeddings  # lazy import

        provider, kwargs = self._embeddings_args(connection, model_name, **params)
        return init_embeddings(model_name, provider=provider, **kwargs)

    def _chat_args(
        self, connection: ConnectionConfig, model_name: str, **params: Any
    ) -> tuple[str, dict[str, Any]]:
        """Resolve (langchain provider string, constructor kwargs) for a chat model. No SDK import."""
        provider = _PROVIDER_TO_LANGCHAIN[connection.provider]
        kwargs = self._connection_kwargs(connection)
        if connection.provider == Provider.AZURE_OPENAI:
            # Azure keys off the *deployment* name; default it to the model name unless overridden.
            kwargs.setdefault("azure_deployment", model_name)
        kwargs.update(params)
        return provider, kwargs

    def _embeddings_args(
        self, connection: ConnectionConfig, model_name: str, **params: Any
    ) -> tuple[str, dict[str, Any]]:
        """Resolve (langchain provider string, constructor kwargs) for an embeddings model."""
        if connection.provider not in _PROVIDER_TO_LANGCHAIN_EMBEDDINGS:
            raise ValueError(f"embeddings not supported for provider {connection.provider}")
        provider = _PROVIDER_TO_LANGCHAIN_EMBEDDINGS[connection.provider]
        kwargs = self._connection_kwargs(connection)
        if connection.provider == Provider.AZURE_OPENAI:
            kwargs.setdefault("azure_deployment", model_name)
        kwargs.update(params)
        return provider, kwargs

    @staticmethod
    def _connection_kwargs(connection: ConnectionConfig) -> dict[str, Any]:
        """Translate connection fields into provider-specific constructor kwargs."""
        kwargs: dict[str, Any] = {}
        if connection.provider == Provider.OPENAI:
            if connection.api_key:
                kwargs["api_key"] = connection.api_key
            if connection.base_url:
                kwargs["base_url"] = connection.base_url
        elif connection.provider == Provider.AZURE_OPENAI:
            if connection.api_key:
                kwargs["api_key"] = connection.api_key
            if connection.base_url:  # reused as the Azure resource endpoint
                kwargs["azure_endpoint"] = connection.base_url
            if connection.api_version:
                kwargs["api_version"] = connection.api_version
        elif connection.provider == Provider.BEDROCK:
            if connection.region:
                kwargs["region_name"] = connection.region
            if connection.auth_profile:
                kwargs["credentials_profile_name"] = connection.auth_profile
        elif connection.provider == Provider.OLLAMA:
            if connection.base_url:
                kwargs["base_url"] = connection.base_url
        # Any explicit passthrough kwargs win.
        kwargs.update(connection.extra or {})
        return kwargs
