"""Tests for the workflow engine (fake model / rag client / tools)."""

from __future__ import annotations

from ai_agent_chat.tools import FunctionTool, ToolResult
from ai_agent_chat.workflows import WorkflowEngine


class _Out:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeModel:
    def invoke(self, messages):
        last = getattr(messages[-1], "content", "")
        return _Out(f"[llm:{last}]")


class _FakeRag:
    def retrieve(self, collection, query, k):
        return [{"text": f"hit for {query} in {collection}"}]


def _engine() -> WorkflowEngine:
    web = FunctionTool(
        "web", "web", {"type": "object", "properties": {}}, lambda **k: ToolResult("web result")
    )
    return WorkflowEngine(
        model_provider=lambda c, m, p: _FakeModel(), rag_client=_FakeRag(), tools={"web": web}
    )


def test_llm_steps_with_templating_and_output() -> None:
    definition = {
        "steps": [
            {"id": "s1", "type": "llm", "prompt": "Topic: {{inputs.topic}}"},
            {"id": "s2", "type": "llm", "prompt": "Expand: {{s1}}"},
        ],
        "output": "{{s2}}",
    }
    result = _engine().run(definition, {"topic": "leave"})
    assert result["status"] == "completed"
    assert "leave" in result["output"]
    assert len(result["trace"]) == 2


def test_rag_step() -> None:
    definition = {
        "steps": [{"id": "r", "type": "rag", "collection": "kb", "query": "{{inputs.q}}"}]
    }
    result = _engine().run(definition, {"q": "policy"})
    assert "hit for policy in kb" in result["output"]


def test_tool_step_with_templated_args() -> None:
    definition = {
        "steps": [{"id": "t", "type": "tool", "tool": "web", "args": {"query": "{{inputs.q}}"}}]
    }
    result = _engine().run(definition, {"q": "news"})
    assert result["output"] == "web result"


def test_unknown_step_fails_gracefully() -> None:
    definition = {"steps": [{"id": "bad", "type": "nonsense"}]}
    result = _engine().run(definition, {})
    assert result["status"] == "failed"
    assert "error" in result["trace"][0]


def test_missing_rag_client_errors() -> None:
    engine = WorkflowEngine(model_provider=lambda c, m, p: _FakeModel(), rag_client=None)
    result = engine.run(
        {"steps": [{"id": "r", "type": "rag", "collection": "kb", "query": "x"}]}, {}
    )
    assert result["status"] == "failed"
