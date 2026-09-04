# Build context is the workspace root (ai-agent/). See deploy/docker-compose.yml.
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1 \
    MCP_HOST=0.0.0.0 \
    MCP_PORT=8090

COPY . .
RUN uv sync --package ai-agent-mcp-web-search --no-dev

EXPOSE 8090
RUN useradd --create-home --uid 10001 appuser && chown -R appuser /app
USER appuser

CMD ["uv", "run", "--no-dev", "--package", "ai-agent-mcp-web-search", \
     "python", "-m", "ai_agent_mcp_web_search"]
