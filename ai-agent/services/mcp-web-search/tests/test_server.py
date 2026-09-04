"""MCP-layer tests: tools are discoverable and callable through the FastMCP server (in-process)."""

from __future__ import annotations

from ai_agent_mcp_web_search.config import McpSettings
from ai_agent_mcp_web_search.fetch import FetchResult
from ai_agent_mcp_web_search.providers import SearchResult, StaticProvider
from ai_agent_mcp_web_search.server import build_server
from ai_agent_mcp_web_search.tools import WebSearchTools


def _server():
    tools = WebSearchTools(
        McpSettings(),
        providers=[StaticProvider(results=[SearchResult("T", "https://e/1", "snip", "static")])],
        fetcher=lambda url, max_chars: FetchResult(url, 200, "text/html", "page text", False),
    )
    return build_server(McpSettings(), tools=tools)


async def test_list_tools_exposes_all_three() -> None:
    server = _server()
    tools = await server.list_tools()
    names = {t.name for t in tools}
    assert {"web_search", "news_search", "fetch_url"} <= names
    # Each tool ships model-facing "when to use" guidance.
    assert all(t.description for t in tools)


async def test_call_web_search_through_mcp() -> None:
    server = _server()
    result = await server.call_tool("web_search", {"query": "hello", "max_results": 2})
    assert result is not None
    assert "https://e/1" in str(result)
