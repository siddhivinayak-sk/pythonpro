# Chat front end (React + Vite + TypeScript)

Phase 0 skeleton of the conversation UI. It fetches `/v1/models` from the chat backend and renders the
model picker across all configured connections. The full experience (streaming chat, history, settings,
vision, RAG/MCP selection, workflows) is built in Phase 3 — see
[`docs/subprojects/conversation-frontend.md`](../../../docs/subprojects/conversation-frontend.md).

## Develop

```bash
npm install
npm run dev        # http://localhost:3000 (proxies /v1 -> http://localhost:8080)
```

Run the chat backend alongside it:

```bash
uv run --package ai-agent-chat uvicorn ai_agent_chat.app:app --reload --port 8080
```

## Build

```bash
npm run build      # type-check + production bundle in dist/
```

The provided `Dockerfile` builds the bundle and serves it via nginx, proxying `/v1` to the `chat-api`
service on the Compose network.
