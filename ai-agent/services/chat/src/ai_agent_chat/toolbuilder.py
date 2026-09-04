"""Build the agent's tool set from a conversation's effective settings.

- ``rag_collections`` (+ a configured RAG service) → a retrieval tool.
- ``mcp_servers`` (list of MCP server URLs) → their discovered tools.

Injectable in the app so tests can supply fake tools; the default connects to real services.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ai_agent_core import get_logger

from .config import ChatSettings
from .mcp_client import McpToolset
from .rag_client import RagClient, RagRetrievalTool
from .tools import AgentTool

log = get_logger("chat-tools")

ToolBuilder = Callable[[ChatSettings, dict[str, Any]], list[AgentTool]]


def default_tool_builder(settings: ChatSettings, effective: dict[str, Any]) -> list[AgentTool]:
    tools: list[AgentTool] = []

    collections = effective.get("rag_collections") or []
    if collections and settings.rag_api_base_url:
        tools.append(RagRetrievalTool(RagClient(settings.rag_api_base_url), collections))

    for server_url in effective.get("mcp_servers") or []:
        try:
            tools.extend(McpToolset(server_url).as_agent_tools())
        except Exception as exc:  # noqa: BLE001 - a down MCP server shouldn't break chat
            log.warning("mcp_tools_unavailable", server=server_url, error=str(exc))

    return tools
