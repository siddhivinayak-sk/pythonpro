"""Agent tool abstraction.

A tool exposes a name, a description (guides the model), a JSON schema for its arguments, and a ``run``
that returns a ``ToolResult`` (text for the model + structured citations for the UI). Tools are converted
to OpenAI-style function specs for ``bind_tools``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


@dataclass
class ToolResult:
    text: str
    citations: list[dict[str, Any]] = field(default_factory=list)


@runtime_checkable
class AgentTool(Protocol):
    name: str
    description: str

    def json_schema(self) -> dict[str, Any]: ...

    def run(self, **kwargs: Any) -> ToolResult: ...


def to_openai_spec(tool: AgentTool) -> dict[str, Any]:
    """OpenAI-style function tool spec accepted by LangChain ``bind_tools`` across providers."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.json_schema(),
        },
    }


class FunctionTool:
    """Adapt a plain callable into an ``AgentTool`` (used for MCP tools and tests)."""

    def __init__(
        self,
        name: str,
        description: str,
        schema: dict[str, Any],
        fn: Callable[..., ToolResult | str],
    ) -> None:
        self.name = name
        self.description = description
        self._schema = schema
        self._fn = fn

    def json_schema(self) -> dict[str, Any]:
        return self._schema

    def run(self, **kwargs: Any) -> ToolResult:
        result = self._fn(**kwargs)
        return result if isinstance(result, ToolResult) else ToolResult(text=str(result))
