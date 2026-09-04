"""Offline tests for the model factory: provider mappings + connection/arg building (no langchain)."""

from __future__ import annotations

import pytest
from ai_agent_core.config.models import ConnectionConfig, Provider
from ai_agent_core.llm.factory import (
    _PROVIDER_TO_LANGCHAIN,
    _PROVIDER_TO_LANGCHAIN_EMBEDDINGS,
    LangChainModelFactory,
)


def _conn(**fields) -> ConnectionConfig:
    return ConnectionConfig(id="c", **fields)


def test_every_provider_is_mapped_for_chat_and_embeddings() -> None:
    for provider in Provider:
        assert provider in _PROVIDER_TO_LANGCHAIN
        assert provider in _PROVIDER_TO_LANGCHAIN_EMBEDDINGS


def test_azure_and_bedrock_provider_strings() -> None:
    assert _PROVIDER_TO_LANGCHAIN[Provider.AZURE_OPENAI] == "azure_openai"
    assert _PROVIDER_TO_LANGCHAIN_EMBEDDINGS[Provider.AZURE_OPENAI] == "azure_openai"
    # Bedrock intentionally differs between chat and embeddings.
    assert _PROVIDER_TO_LANGCHAIN[Provider.BEDROCK] == "bedrock_converse"
    assert _PROVIDER_TO_LANGCHAIN_EMBEDDINGS[Provider.BEDROCK] == "bedrock"


def test_openai_connection_kwargs() -> None:
    kw = LangChainModelFactory._connection_kwargs(
        _conn(provider=Provider.OPENAI, api_key="sk", base_url="https://gw")
    )
    assert kw == {"api_key": "sk", "base_url": "https://gw"}


def test_azure_connection_kwargs_maps_endpoint_and_version() -> None:
    kw = LangChainModelFactory._connection_kwargs(
        _conn(
            provider=Provider.AZURE_OPENAI,
            api_key="az",
            base_url="https://r.openai.azure.com/",
            api_version="2024-10-21",
        )
    )
    assert kw == {
        "api_key": "az",
        "azure_endpoint": "https://r.openai.azure.com/",
        "api_version": "2024-10-21",
    }


def test_bedrock_and_ollama_connection_kwargs() -> None:
    bedrock = LangChainModelFactory._connection_kwargs(
        _conn(provider=Provider.BEDROCK, region="us-east-1", auth_profile="default")
    )
    assert bedrock == {"region_name": "us-east-1", "credentials_profile_name": "default"}
    ollama = LangChainModelFactory._connection_kwargs(
        _conn(provider=Provider.OLLAMA, base_url="http://ollama:11434")
    )
    assert ollama == {"base_url": "http://ollama:11434"}


def test_chat_args_defaults_azure_deployment_to_model_name() -> None:
    provider, kwargs = LangChainModelFactory()._chat_args(
        _conn(provider=Provider.AZURE_OPENAI, api_key="az", base_url="https://r", api_version="v"),
        "gpt-4o",
    )
    assert provider == "azure_openai"
    assert kwargs["azure_deployment"] == "gpt-4o"


def test_chat_args_respects_explicit_azure_deployment() -> None:
    _, kwargs = LangChainModelFactory()._chat_args(
        _conn(provider=Provider.AZURE_OPENAI, api_key="az", extra={"azure_deployment": "dep-x"}),
        "gpt-4o",
    )
    assert kwargs["azure_deployment"] == "dep-x"


def test_embeddings_args_for_azure() -> None:
    provider, kwargs = LangChainModelFactory()._embeddings_args(
        _conn(provider=Provider.AZURE_OPENAI, api_key="az", base_url="https://r", api_version="v"),
        "text-embedding-3-large",
    )
    assert provider == "azure_openai"
    assert kwargs["azure_deployment"] == "text-embedding-3-large"
    assert kwargs["azure_endpoint"] == "https://r"


def test_params_override_connection_kwargs() -> None:
    _, kwargs = LangChainModelFactory()._chat_args(
        _conn(provider=Provider.OPENAI, api_key="sk"), "gpt-4o", temperature=0.3
    )
    assert kwargs["temperature"] == 0.3


def test_embeddings_unsupported_provider_raises() -> None:
    # Force an unmapped provider value to exercise the guard.
    conn = _conn(provider=Provider.OLLAMA)
    object.__setattr__(conn, "provider", "made-up")
    with pytest.raises(ValueError, match="embeddings not supported"):
        LangChainModelFactory()._embeddings_args(conn, "m")
