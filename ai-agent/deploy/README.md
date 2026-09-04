# Deployment (`deploy/`)

Docker Compose orchestration for the AI-Agent Platform. Every sub-project is isolated behind a **profile**
so you run only what you need.

## Prerequisites
- Docker + Docker Compose
- `cp .env.example .env` and edit secrets (Postgres password, provider keys, Keycloak admin)

## Profiles

| Profile | Brings up | Published ports |
|---------|-----------|-----------------|
| `chat` | chat-api + chat-frontend | 8080 (api), 3000 (ui) |
| `rag` | rag-api + rag-indexer + pgvector | 8081 (api), 5432 (db) |
| `mcp` | mcp-web-search + searxng | 8090 (mcp), 8088 (searxng) |
| `eval` | evaluation-runner (on demand) | — |
| `pgvector` | pgvector only | 5432 |
| `ollama` | Ollama runtime | 11434 |
| `keycloak` | Keycloak (dev) | 8083 |
| `observability` | Prometheus + Grafana + Langfuse | 9090, 3002, 3001 |
| `all` | everything above | all |

The chat (`:8080`) and RAG (`:8081`) services expose Prometheus metrics at `/metrics`; Prometheus scrapes
them via `prometheus/prometheus.yml`. See [`docs/OPERATIONS.md`](../docs/OPERATIONS.md).

## Examples

```bash
# MCP web search only
docker compose --profile mcp up

# RAG (includes pgvector)
docker compose --profile rag up

# Chat backend + UI (point it at existing rag/mcp if running)
docker compose --profile chat up

# Run an evaluation on demand
docker compose --profile eval run --rm evaluation-runner run --suite rag

# Everything
docker compose --profile all up
```

## Validate the compose file (no build)

```bash
docker compose config
```

## Notes
- Python service images build from the **workspace root** context (`..`) so they can install
  `ai-agent-core` from the uv workspace.
- The chat backend is pointed at `services/chat/config/llm.example.yaml`, which declares OpenAI, Bedrock,
  and Ollama connections — every chat model shows up in `GET /v1/models`.
- **Security:** these defaults are for local/dev. Before any shared/prod deployment, set real secrets,
  put services behind a TLS reverse proxy, and enable auth on the chat/RAG/MCP endpoints (see the design
  docs' security sections).
