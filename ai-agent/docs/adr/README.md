# Architecture Decision Records (ADRs)

Short, dated records of significant, hard-to-reverse decisions. Each ADR states the context, the decision,
and its consequences. Supersede rather than edit once accepted.

| ADR | Title | Status |
|-----|-------|--------|
| [0001](./0001-monorepo-uv-workspace.md) | Monorepo managed as a uv workspace | Accepted |
| [0002](./0002-llm-connection-registry.md) | Multi-connection LLM registry via `init_chat_model` | Accepted |
| [0003](./0003-pgvector-default-vector-store.md) | pgvector as the default vector store | Accepted |
| [0004](./0004-react-fastapi-frontend.md) | Custom React + FastAPI conversation front end | Accepted |
| [0005](./0005-mcp-streamable-http-searxng.md) | MCP over Streamable HTTP with SearXNG default | Accepted |
| [0006](./0006-testable-core-lazy-adapters.md) | Dependency-free core with lazy production adapters | Accepted |

Open questions that may become future ADRs are listed in
[nfr-governance.md §13](../nfr-governance.md#13-assumptions-constraints--open-questions).
