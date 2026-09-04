# ADR 0002 — Multi-connection LLM registry via `init_chat_model`

- **Status:** Accepted (2026-09)
- **Context:** The platform must support OpenAI, Bedrock, and Ollama — any one or all — with *multiple*
  named connections per provider, and let users pick a model/connection at runtime.
- **Decision:** A configuration-driven **connection registry** (`ai_agent_core.llm`) resolves models
  across all enabled connections and constructs chat models through LangChain's provider-agnostic
  `init_chat_model` (a small factory maps our `Provider` enum to `model_provider`). Provider SDKs are
  imported lazily.
- **Consequences:** Adding a provider is a small adapter change; consumers are unchanged. Credentials stay
  server-side; the UI receives only non-secret descriptors. Live model discovery (e.g., Ollama `/api/tags`)
  is a later enhancement — for now models are declared/allowlisted in config.
