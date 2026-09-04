"""Tests for configuration models and the layered loader."""

from __future__ import annotations

import textwrap

import pytest
from ai_agent_core.config import (
    Capability,
    ConnectionConfig,
    LLMConfig,
    ModelDescriptor,
    Provider,
)
from ai_agent_core.config.base import load_config, load_yaml


def test_model_descriptor_defaults() -> None:
    m = ModelDescriptor(name="gpt-4o")
    assert m.capabilities == [Capability.CHAT]
    assert m.label == "gpt-4o"
    assert m.has(Capability.CHAT)
    assert not m.has(Capability.VISION)


def test_model_descriptor_label_prefers_display_name() -> None:
    m = ModelDescriptor(name="gpt-4o", display_name="GPT-4o (vision)")
    assert m.label == "GPT-4o (vision)"


def test_llm_config_parses_nested_structure() -> None:
    cfg = LLMConfig.model_validate(
        {
            "default_connection": "ollama-local",
            "connections": [
                {
                    "id": "ollama-local",
                    "provider": "ollama",
                    "base_url": "http://localhost:11434",
                    "models": [{"name": "llama3.1", "capabilities": ["chat", "tools"]}],
                }
            ],
        }
    )
    assert cfg.default_connection == "ollama-local"
    assert len(cfg.connections) == 1
    conn = cfg.connections[0]
    assert conn.provider is Provider.OLLAMA
    assert conn.enabled is True
    assert conn.models[0].has(Capability.TOOLS)


def test_load_yaml_missing_file_returns_empty(tmp_path) -> None:
    assert load_yaml(tmp_path / "nope.yaml") == {}


def test_load_yaml_rejects_non_mapping(tmp_path) -> None:
    p = tmp_path / "bad.yaml"
    p.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_yaml(p)


def test_load_config_expands_env(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("MY_OPENAI_KEY", "sk-secret-123")
    cfg_file = tmp_path / "llm.yaml"
    cfg_file.write_text(
        textwrap.dedent(
            """
            default_connection: openai-prod
            connections:
              - id: openai-prod
                provider: openai
                api_key: ${MY_OPENAI_KEY}
                models:
                  - name: gpt-4o
                    capabilities: [chat, vision, tools]
            """
        ),
        encoding="utf-8",
    )
    cfg = load_config(LLMConfig, cfg_file)
    assert cfg.connections[0].api_key == "sk-secret-123"
    assert Capability.VISION in cfg.connections[0].models[0].capabilities


def test_load_config_no_path_yields_defaults() -> None:
    cfg = load_config(LLMConfig, None)
    assert cfg.connections == []
    assert cfg.default_connection is None


def test_connection_config_defaults() -> None:
    c = ConnectionConfig(id="x", provider=Provider.OPENAI)
    assert c.enabled is True
    assert c.models == []
    assert c.extra == {}
