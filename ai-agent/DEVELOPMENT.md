# Development guide

This is the developer setup for the **AI-Agent Platform** monorepo. For the design/analysis, see
[`docs/comprehensive-analysis.md`](docs/comprehensive-analysis.md).

> **Phase 0 scaffold.** This repo currently contains the shared core library and runnable *skeletons* of
> each service (health endpoints, config wiring, and the LLM connection registry). Full feature
> implementation follows the phased roadmap.

## Prerequisites

- Python 3.12+ (3.13 recommended)
- [uv](https://docs.astral.sh/uv/) for Python dependency management
- Node 20+ (for the chat front end)
- Docker + Docker Compose (for containerized runs)

## Workspace layout

```
packages/ai_agent_core   # shared library (config, LLM registry, telemetry, schemas)
services/rag             # RAG retrieval API + indexer (FastAPI)
services/chat            # conversation backend (FastAPI) + frontend/ (React+Vite+TS)
services/mcp-web-search  # MCP web-search server (FastMCP)
services/evaluation      # evaluation CLI (RAGAS/DeepEval)
deploy/                  # docker-compose + env templates
```

The repo is a single **uv workspace**; all Python members share one lockfile and virtual environment.

## Common commands

```bash
# Create the venv and install all workspace members + dev tools
uv sync

# Run the shared-core tests
uv run pytest packages/ai_agent_core

# Run all tests
uv run pytest

# Lint & type-check
uv run ruff check .
uv run mypy packages services

# Run a service locally (examples)
uv run uvicorn ai_agent_rag.app:app --reload --port 8081
uv run uvicorn ai_agent_chat.app:app --reload --port 8080
uv run python -m ai_agent_mcp_web_search
uv run python -m ai_agent_eval run --suite rag

# Front end
cd services/chat/frontend && npm install && npm run dev
```

## Containerized run (profiles)

```bash
# From deploy/: run only what you need
docker compose --profile mcp up            # MCP web search + SearXNG
docker compose --profile rag --profile pgvector up
docker compose --profile all up            # everything
```

Copy the env template before running: `cp deploy/.env.example deploy/.env` and fill in values.

## Conventions

- Config is layered: env vars > `.env` > YAML config file > defaults (see `ai_agent_core.config`).
- Never commit secrets; use `deploy/.env` (git-ignored) or a secret manager.
- Each service depends on `ai_agent_core` via the workspace; keep cross-cutting logic there.
