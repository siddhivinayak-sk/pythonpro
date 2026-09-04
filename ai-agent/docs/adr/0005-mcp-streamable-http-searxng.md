# ADR 0005 — MCP over Streamable HTTP with SearXNG default

- **Status:** Accepted (2026-09)
- **Context:** Web grounding should be exposed through a standard, discoverable interface and use
  open-source search. MCP transports: stdio (local) vs Streamable HTTP (networked); HTTP+SSE is
  deprecated (2025-03-26).
- **Decision:** Implement a **FastMCP** server over **Streamable HTTP** exposing `web_search`,
  `news_search`, and `fetch_url`. Default backend is **SearXNG** (self-hosted, no API key) behind a
  provider abstraction with optional fallbacks; the chat agent consumes it via `langchain-mcp-adapters`.
- **Consequences:** The server runs as a networked service any MCP client can use; SSRF-safe fetching is
  required (implemented). Bearer-token auth/rate limiting are config-reserved and enforced at the edge for
  now. Managed providers (Tavily/Brave) remain optional.
