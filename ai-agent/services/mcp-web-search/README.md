# MCP Web-Search Server (`services/mcp-web-search`)

A Model Context Protocol server exposing web-grounding tools over open-source search (SearXNG). Full
design: [`docs/subprojects/mcp-web-search.md`](../../docs/subprojects/mcp-web-search.md).

> **Status: Phase 1 complete.** SSRF-hardened fetching, readability extraction, TTL caching, a pluggable
> provider chain (fallback + RRF merge), and structured tool results are implemented and tested. Bearer-
> token auth and rate limiting are config-reserved for a later hardening pass (treat as a deployment
> concern behind a reverse proxy for now).

## Tools

| Tool | Signature | Purpose |
|------|-----------|---------|
| `web_search` | `(query, max_results=5, time_range=None)` | Ranked web results (`title, url, snippet, source`) for grounding + citations |
| `news_search` | `(query, max_results=5, time_range="week")` | Recency-biased results (SearXNG `news` category) |
| `fetch_url` | `(url, max_chars=8000)` | Fetch a page and return its readable main text (SSRF-guarded) |

All tools return a JSON envelope; failures come back as a structured `error` field (never a raw
exception), so the agent always gets a usable message.

## Key behaviors

- **SSRF protection** (`fetch.py`): http/https only; the host is resolved and refused if it maps to a
  private / loopback / link-local / reserved / multicast address (blocks `169.254.169.254` and internal
  hosts); redirects are followed manually and re-validated per hop; content-type allowlist + byte/char
  caps. Known limitation: pre-resolution checks don't fully defeat DNS-rebinding — pin the IP or use an
  egress proxy in hostile environments.
- **Readability extraction**: strips scripts/nav/ads and prefers `<main>`/`<article>` (BeautifulSoup;
  install the `extraction` extra for trafilatura-quality output).
- **Provider chain**: `fallback` (first non-empty) or `merge` (Reciprocal Rank Fusion across providers).
  SearXNG by default; DuckDuckGo optional (`duckduckgo` extra).
- **Caching**: in-memory TTL cache for search results (`MCP_CACHE_TTL_SECONDS`).

## Run locally

Requires a SearXNG instance (see `deploy/` for a Compose profile), then:

```powershell
uv sync --all-packages
$env:MCP_SEARXNG_BASE_URL = "http://localhost:8088"
uv run --package ai-agent-mcp-web-search python -m ai_agent_mcp_web_search
# Streamable HTTP endpoint: http://localhost:8090/mcp
```

Or just run the whole MCP profile (server + SearXNG): `docker compose --profile mcp up` from `deploy/`.

## Configuration (env, prefix `MCP_`)

| Var | Default | Notes |
|-----|---------|-------|
| `MCP_SEARXNG_BASE_URL` | `http://searxng:8080` | SearXNG JSON API base |
| `MCP_PROVIDER_ORDER` | `["searxng"]` | JSON list; e.g. `["searxng","duckduckgo"]` |
| `MCP_SEARCH_MODE` | `fallback` | `fallback` or `merge` (RRF) |
| `MCP_MAX_RESULTS` | `5` | Cap per search |
| `MCP_CACHE_TTL_SECONDS` | `300` | Search cache TTL |
| `MCP_MAX_REDIRECTS` | `3` | Redirect hops (each re-validated) |
| `MCP_ALLOW_PRIVATE_NETWORKS` | `false` | Keep false except trusted internal use |
| `MCP_FETCH_MAX_CHARS` / `MCP_FETCH_MAX_BYTES` | `8000` / `2000000` | Extraction/size caps |

## Use from the chat agent

Register the server in the chat backend's MCP config (`url: http://mcp-web-search:8090/mcp`, transport
`streamable_http`). The agent discovers and calls the tools via `langchain-mcp-adapters`.

## Tests

```bash
uv run pytest services/mcp-web-search
```

Covers SSRF validation, redirect re-validation, extraction, cache TTL/LRU, RRF merge, provider fallback,
the tools layer, and discovery/invocation through the MCP server (`list_tools`/`call_tool`).
