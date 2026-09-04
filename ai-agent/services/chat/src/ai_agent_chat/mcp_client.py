"""MCP integration: discover and call tools on configured MCP servers, exposed as agent tools.

The real client uses the official MCP streamable-HTTP client (lazy import, one short-lived session per
call). Like the pgvector adapter, this path needs a live MCP server (the ``mcp`` Docker profile) and is
not exercised by the in-process test suite; the citation-parsing helper is unit-tested on its own.
"""

from __future__ import annotations

import json
from typing import Any

from .tools import FunctionTool, ToolResult


def mcp_result_to_tool_result(raw: str) -> ToolResult:
    """Wrap an MCP tool's text output, extracting web citations when the payload looks like search results."""
    citations: list[dict[str, Any]] = []
    try:
        data = json.loads(raw)
    except (ValueError, TypeError):
        return ToolResult(text=raw)
    results = data.get("results") if isinstance(data, dict) else None
    if isinstance(results, list):
        for item in results:
            if isinstance(item, dict) and item.get("url"):
                citations.append({"url": item["url"], "title": item.get("title"), "source": "web"})
    return ToolResult(text=raw, citations=citations)


class McpToolset:
    """Connect to one MCP server (streamable HTTP), list its tools, and invoke them."""

    def __init__(self, server_url: str, timeout: int = 30) -> None:
        self.server_url = server_url
        self.timeout = timeout

    def _run(self, coro):
        import asyncio

        return asyncio.run(coro)

    async def _list(self) -> list[dict[str, Any]]:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        async with (
            streamablehttp_client(self.server_url) as (read, write, _),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            resp = await session.list_tools()
            return [
                {
                    "name": t.name,
                    "description": t.description or "",
                    "schema": t.inputSchema or {"type": "object", "properties": {}},
                }
                for t in resp.tools
            ]

    async def _call(self, name: str, arguments: dict[str, Any]) -> str:
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client

        async with (
            streamablehttp_client(self.server_url) as (read, write, _),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            result = await session.call_tool(name, arguments)
            parts = [getattr(block, "text", "") for block in result.content]
            return "\n".join(p for p in parts if p)

    def list_tools(self) -> list[dict[str, Any]]:
        return self._run(self._list())

    def call_tool(self, name: str, arguments: dict[str, Any]) -> str:
        return self._run(self._call(name, arguments))

    def as_agent_tools(self) -> list[FunctionTool]:
        tools: list[FunctionTool] = []
        for spec in self.list_tools():
            tools.append(self._wrap(spec))
        return tools

    def _wrap(self, spec: dict[str, Any]) -> FunctionTool:
        def fn(**kwargs: Any) -> ToolResult:
            return mcp_result_to_tool_result(self.call_tool(spec["name"], kwargs))

        return FunctionTool(spec["name"], spec["description"], spec["schema"], fn)
