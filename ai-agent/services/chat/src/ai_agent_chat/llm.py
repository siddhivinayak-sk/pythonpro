"""Build the LLM connection registry for the chat service.

This is where the multi-provider / multi-connection requirement becomes concrete: a config file can
declare many OpenAI/Bedrock/Ollama connections, and every chat model they expose is returned to the UI's
model picker.
"""

from __future__ import annotations

from ai_agent_core import (
    Capability,
    ConnectionConfig,
    LLMConfig,
    LLMConnectionRegistry,
    ModelDescriptor,
    Provider,
    load_config,
)

from .config import ChatSettings


def _default_config() -> LLMConfig:
    """A ready-to-run default: a single local Ollama connection."""
    return LLMConfig(
        default_connection="ollama-local",
        connections=[
            ConnectionConfig(
                id="ollama-local",
                provider=Provider.OLLAMA,
                base_url="http://localhost:11434",
                models=[
                    ModelDescriptor(
                        name="llama3.1",
                        display_name="Llama 3.1 (local)",
                        capabilities=[Capability.CHAT, Capability.TOOLS],
                    )
                ],
            )
        ],
    )


def build_registry(settings: ChatSettings) -> LLMConnectionRegistry:
    config = (
        load_config(LLMConfig, settings.llm_config_file)
        if settings.llm_config_file
        else _default_config()
    )
    return LLMConnectionRegistry.from_config(config)
