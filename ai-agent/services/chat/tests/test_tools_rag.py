"""Tests for the tool abstraction, the RAG retrieval tool, and MCP result parsing."""

from __future__ import annotations

from ai_agent_chat.mcp_client import mcp_result_to_tool_result
from ai_agent_chat.rag_client import RagRetrievalTool
from ai_agent_chat.tools import FunctionTool, ToolResult, to_openai_spec


class _FakeRag:
    def retrieve(self, collection, query, k):
        return [
            {
                "text": "Employees receive parental leave.",
                "score": 0.9,
                "citation": {"path": "hr/leave.pdf", "title": "Leave"},
            },
            {
                "text": "Vacation accrues monthly.",
                "score": 0.5,
                "citation": {"path": "hr/vacation.pdf"},
            },
        ]


def test_rag_tool_returns_text_and_citations() -> None:
    tool = RagRetrievalTool(_FakeRag(), ["kb"], k=5)
    result = tool.run(query="parental leave")
    assert "parental leave" in result.text.lower()
    assert result.citations[0]["path"] == "hr/leave.pdf"
    assert result.citations[0]["source"] == "rag"


def test_rag_tool_no_hits() -> None:
    class _Empty:
        def retrieve(self, c, q, k):
            return []

    assert RagRetrievalTool(_Empty(), ["kb"]).run(query="x").text == "No relevant passages found."


def test_to_openai_spec_shape() -> None:
    tool = RagRetrievalTool(_FakeRag(), ["kb"])
    spec = to_openai_spec(tool)
    assert spec["type"] == "function"
    assert spec["function"]["name"] == "retrieve"
    assert "query" in spec["function"]["parameters"]["properties"]


def test_function_tool_wraps_string_result() -> None:
    tool = FunctionTool("echo", "echo", {"type": "object", "properties": {}}, lambda **k: "hi")
    assert tool.run().text == "hi"


def test_function_tool_passthrough_toolresult() -> None:
    tr = ToolResult(text="x", citations=[{"url": "u"}])
    tool = FunctionTool("t", "t", {"type": "object", "properties": {}}, lambda **k: tr)
    assert tool.run().citations == [{"url": "u"}]


def test_mcp_result_parses_web_citations() -> None:
    raw = (
        '{"results": [{"title": "T", "url": "https://e/1"}, {"title": "N", "url": "https://e/2"}]}'
    )
    result = mcp_result_to_tool_result(raw)
    assert {c["url"] for c in result.citations} == {"https://e/1", "https://e/2"}
    assert all(c["source"] == "web" for c in result.citations)


def test_mcp_result_non_json_is_plain_text() -> None:
    result = mcp_result_to_tool_result("just text")
    assert result.text == "just text"
    assert result.citations == []
