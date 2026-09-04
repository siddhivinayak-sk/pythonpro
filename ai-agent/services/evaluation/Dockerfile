# Build context is the workspace root (ai-agent/). See deploy/docker-compose.yml.
# Typically run on demand: `docker compose --profile eval run --rm evaluation-runner run --suite rag`
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

COPY . .
RUN uv sync --package ai-agent-evaluation --no-dev

WORKDIR /app/services/evaluation
ENTRYPOINT ["uv", "run", "--no-dev", "--package", "ai-agent-evaluation", "python", "-m", "ai_agent_eval"]
CMD ["run", "--suite", "rag"]
