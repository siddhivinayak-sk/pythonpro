"""FastMCP web-search server (Streamable HTTP).

Thin MCP wiring over the tool logic in ``tools.py``. ``mcp`` is imported lazily so importing this module
does not require the SDK, and ``build_server`` accepts an injected ``WebSearchTools`` for testing.
"""

from __future__ import annotations

from typing import Any

from ai_agent_core import configure_logging, get_logger

from .config import McpSettings
from .tools import WebSearchTools


def build_server(settings: McpSettings | None = None, tools: WebSearchTools | None = None) -> Any:
    """Construct and return a configured FastMCP server (does not start it)."""
    from mcp.server.fastmcp import FastMCP  # lazy import

    settings = settings or McpSettings()
    configure_logging(level=settings.log_level, json_logs=settings.log_json)
    log = get_logger("mcp-web-search")
    tools = tools or WebSearchTools(settings)

    mcp = FastMCP(
        settings.service_name,
        host=settings.host,
        port=settings.port,
        streamable_http_path=settings.path,
    )

    @mcp.tool()
    def web_search(query: str, max_results: int = 5, time_range: str | None = None) -> dict:
        """Search the web for current information.

        Returns ranked results (title, url, snippet, source). Use this to ground answers in up-to-date
        web content and to find sources to cite. ``time_range`` may be day/week/month/year.
        """
        return tools.web_search(query, max_results=max_results, time_range=time_range)

    @mcp.tool()
    def news_search(query: str, max_results: int = 5, time_range: str = "week") -> dict:
        """Search recent news for a topic. Prefer this over web_search when recency matters."""
        return tools.news_search(query, max_results=max_results, time_range=time_range)

    @mcp.tool()
    def fetch_url(url: str, max_chars: int = 8000) -> dict:
        """Fetch a URL and return its readable main text (truncated to ``max_chars``).

        Use after ``web_search`` to read a specific page for fuller context. Only http/https public URLs
        are allowed; internal/private addresses are refused for safety.
        """
        return tools.fetch_url(url, max_chars=max_chars)

    log.info(
        "mcp_web_search_ready",
        providers=[p.name for p in tools.providers],
        search_mode=settings.search_mode,
        port=settings.port,
    )
    return mcp


def main() -> None:
    settings = McpSettings()
    server = build_server(settings)
    server.run(transport="streamable-http")


if __name__ == "__main__":
    main()
