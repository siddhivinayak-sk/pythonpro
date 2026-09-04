# Build context is the workspace root (ai-agent/). See deploy/docker-compose.yml.
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

COPY . .
RUN uv sync --package ai-agent-chat --no-dev

EXPOSE 8080
RUN useradd --create-home --uid 10001 appuser && chown -R appuser /app
USER appuser

CMD ["uv", "run", "--no-dev", "--package", "ai-agent-chat", \
     "uvicorn", "ai_agent_chat.app:app", "--host", "0.0.0.0", "--port", "8080"]
