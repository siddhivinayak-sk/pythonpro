# Build context is the workspace root (ai-agent/). See deploy/docker-compose.yml.
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

# Copy the workspace and install just this service (+ its workspace deps).
# Install with the pgvector extra so the default production backend works in the compose stack.
COPY . .
RUN uv sync --package ai-agent-rag --extra pgvector --no-dev

EXPOSE 8081
# Non-root runtime user
RUN useradd --create-home --uid 10001 appuser && chown -R appuser /app
USER appuser

CMD ["uv", "run", "--no-dev", "--package", "ai-agent-rag", \
     "uvicorn", "ai_agent_rag.app:app", "--host", "0.0.0.0", "--port", "8081"]
