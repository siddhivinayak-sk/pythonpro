# Conversation Front End (`services/chat`)

Backend (FastAPI) + `frontend/` (React + Vite + TS) for the chat experience. Full design:
[`docs/subprojects/conversation-frontend.md`](../../docs/subprojects/conversation-frontend.md).

> **Status: Phases 3–4 implemented.** Phase 3: auth (local admin + OIDC verifier), streaming chat,
> history + resume, settings, model switching, vision, app DB. Phase 4: a **tool-calling agent** that
> grounds answers with **RAG retrieval + MCP web search** (with citations) and a **workflow engine**
> (llm/rag/tool steps). Also: **ad-hoc context files** attached to a conversation and injected as
> reference data, **assistant image-output rendering**, and **audio STT/TTS endpoints** behind a
> pluggable provider, plus **input/output moderation**, a **response cache** (exact/semantic), and
> optional **Langfuse tracing** — all injectable and off by default. The React UI renders assistant
> **markdown**, has a **settings panel** (account + per-conversation) with RAG/MCP discovery, and
> **light/dark/system** theming. Backend: 79 tests; the React UI
> type-checks and builds. Streaming-with-tools emits
> the grounded answer once; token-level streaming through tools and a full LangGraph multi-agent runtime
> are future work.

## What's implemented

| Area | Details |
|------|---------|
| **Auth** | Local admin (pbkdf2 + server-side sessions) and an OIDC/Keycloak verifier abstraction (JWKS validation, lazy `jose`); Bearer or `session` cookie. |
| **Chat** | Streaming (SSE) and sync turns via an **injectable** chat model (LLM registry by default); memory windowing; per-message model/connection recorded. |
| **History** | Conversations CRUD, auto-title, list ordered by activity, resume with full history. |
| **Settings** | System < user < conversation precedence. Generation params (temperature, top_p, max tokens, frequency/presence penalty, stop, past-messages-included), system prompt, theme, model, RAG/MCP selections. Editable from the **⚙ Settings** panel at either the **account** or **this-conversation** scope; RAG collections + MCP servers are chosen from discovered options. Params are applied to the model call and **filtered per provider** (unsupported keys are dropped, e.g. penalties on Ollama). |
| **Frontend UX** | Assistant replies render **markdown** (headings, lists, code blocks, tables, links, images) safely (React elements, sanitized URLs); **light/dark/system** theme. |
| **Model switching** | `/v1/models` lists all connections' chat models; switch mid-conversation. |
| **Vision** | Image upload + owner-scoped serving; data-URI images passed to vision-capable models. Assistant image output (markdown, `data:` URI, or http image URLs) renders inline in the transcript. |
| **Context files** | Attach a text file to a conversation; stored per-conversation and injected into the prompt as delimited, untrusted **reference data** (truncated to `context_char_cap`). |
| **Audio** | `/v1/audio/transcribe` (STT) and `/v1/audio/speech` (TTS) behind a pluggable `SpeechProvider`; the default `NullSpeechProvider` responds `501` until a real provider is wired (kept out of the default install so no speech model is required). |
| **Moderation** | Injectable `ModerationProvider` (`AllowAll` default; dep-free `Keyword` blocklist; API providers pluggable). Input moderation short-circuits before the model; output moderation replaces flagged answers with a logged message and skips caching. Enable with `CHAT_MODERATION_ENABLED` + `CHAT_MODERATION_BLOCKLIST`. |
| **Response cache** | Injectable `ResponseCache` — `off` (default), `exact` (LRU), or `semantic` (cosine similarity). Set `CHAT_CACHE_MODE`. Semantic mode auto-wires a registry-backed embedder (the first configured embeddings model) and degrades to exact if none is available; it keys on the query (off-by-default, stateless/FAQ-oriented). |
| **Tracing** | Optional Langfuse tracing wired via `ai_agent_core.build_langfuse_callbacks` (opt-in with `CHAT_TRACING_ENABLED` + `LANGFUSE_*`); a no-op when the package/keys are absent. |
| **Database** | SQLAlchemy 2.0 — **SQLite default**, DuckDB optional, configurable location (`:memory:` for tests). |
| **Agent tools** | Tool-calling loop with **RAG retrieval + MCP web-search**; answers carry `citations` + `tools_used`. Auto-routed when a conversation selects RAG collections / MCP servers. |
| **Workflows** | Saved multi-step automations (`llm`/`rag`/`tool` steps, `{{var}}` templating) with run history. |

## Run the backend

```bash
uv sync --all-packages
$env:CHAT_ADMIN_PASSWORD = "change-me"          # PowerShell; enables local admin login
$env:CHAT_DB_PATH = "./data/chat/app.db"
# To actually call models, install a provider extra + point at a config.
# Provider extras: openai | azure-openai | bedrock | ollama | all-providers
#   uv sync --package ai-agent-core --extra ollama
#   $env:CHAT_LLM_CONFIG_FILE = "services/chat/config/llm.example.yaml"
uv run --package ai-agent-chat uvicorn ai_agent_chat.app:app --reload --port 8080
```

## Run the front end

```bash
cd services/chat/frontend
npm install
npm run dev        # http://localhost:3000 (proxies /v1 -> http://localhost:8080)
```

## Key endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/v1/auth/login` · `/v1/auth/oidc/token` · `/v1/auth/logout` · GET `/v1/auth/me` | auth |
| GET | `/v1/models` | chat models across all connections |
| GET/POST | `/v1/conversations` (+ `PATCH`/`DELETE` `/{id}`) | history |
| GET | `/v1/conversations/{id}/messages` | transcript |
| POST | `/v1/conversations/{id}/chat` · `/chat/stream` | chat (sync / SSE) |
| GET/PUT | `/v1/settings` · GET/PUT `/v1/conversations/{id}/settings` | settings (account + per-conversation) |
| GET | `/v1/rag/collections` · `/v1/mcp/servers` | discovery for the settings UI |
| POST/GET | `/v1/uploads` · `/v1/uploads/{id}` | vision images |
| POST/GET/DELETE | `/v1/conversations/{id}/context` (+ `/{context_id}`) | ad-hoc context files |
| POST | `/v1/audio/transcribe` · `/v1/audio/speech` | STT / TTS (pluggable provider) |
| GET/POST | `/v1/workflows` (+ `/{id}`, `/{id}/run`, `/{id}/runs`) | workflows |

## Tests

```bash
uv run pytest services/chat
```

Covers hashing, auth (local + fake OIDC), the store (settings precedence, memory window, context CRUD),
the chat orchestrator (fake model, streaming, vision parts, context injection), and the API end-to-end
(login → chat → switch model → resume → vision upload → context upload → audio with a fake provider and
`501` when unconfigured → logout).
