# ai-agent-core

Shared library for the AI-Agent Platform. Every service depends on it for cross-cutting concerns so the
logic lives in one place.

## Modules

| Module | Purpose |
|--------|---------|
| `ai_agent_core.config` | Typed configuration models (providers, connections) + layered loader (YAML + `${ENV}` expansion). |
| `ai_agent_core.llm` | **LLM connection registry** and model factory — register many OpenAI/Bedrock/Ollama connections and resolve chat models at runtime via `init_chat_model`. |
| `ai_agent_core.telemetry` | Structured logging (structlog) setup. |
| `ai_agent_core.schemas` | Shared DTOs used across service APIs (health, model info, retrieval). |

## Quick example

```python
from ai_agent_core import LLMConfig, LLMConnectionRegistry, configure_logging

configure_logging(level="INFO")

config = LLMConfig.model_validate(
    {
        "default_connection": "ollama-local",
        "connections": [
            {
                "id": "ollama-local",
                "provider": "ollama",
                "base_url": "http://localhost:11434",
                "models": [{"name": "llama3.1", "capabilities": ["chat", "tools"]}],
            },
            {
                "id": "openai-prod",
                "provider": "openai",
                "api_key": "${OPENAI_API_KEY}",
                "models": [{"name": "gpt-4o", "capabilities": ["chat", "vision", "tools"]}],
            },
        ],
    }
)

registry = LLMConnectionRegistry.from_config(config)
for m in registry.list_chat_models():
    print(m.connection_id, m.model_name, m.capabilities)

# Requires the relevant provider extra installed (e.g. pip install "ai-agent-core[ollama]")
# model = registry.get_chat_model("ollama-local", "llama3.1", temperature=0.2)
```

The factory imports LangChain lazily, so importing this package and running its tests does **not** require
any provider SDK.
