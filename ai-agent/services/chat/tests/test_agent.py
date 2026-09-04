"""Tests for the tool-calling agent orchestrator (fake model + fake tools)."""

from __future__ import annotations

from ai_agent_chat.agent import AgentOrchestrator, ModelTurn, ToolInvocation
from ai_agent_chat.config import ChatSettings
from ai_agent_chat.db import Database
from ai_agent_chat.llm import build_registry
from ai_agent_chat.store import ChatStore
from ai_agent_chat.tools import FunctionTool, ToolResult


class _FakeAgentModel:
    def __init__(self, script: list[ModelTurn]) -> None:
        self._script = script
        self._i = 0
        self.bound = False

    def bind_tools(self, specs):
        self.bound = True
        return self

    def invoke(self, messages):
        turn = self._script[min(self._i, len(self._script) - 1)]
        self._i += 1
        return turn


def _agent(script, *, max_iterations=4):
    settings = ChatSettings(db_path=":memory:")
    store = ChatStore(Database(settings), settings)
    registry = build_registry(settings)
    orch = AgentOrchestrator(
        registry,
        store,
        settings,
        model_provider=lambda c, m, p: _FakeAgentModel(script),
        max_iterations=max_iterations,
    )
    return orch, store


def _rag_tool():
    return FunctionTool(
        "retrieve",
        "retrieve",
        {"type": "object", "properties": {"query": {"type": "string"}}},
        lambda query: ToolResult(
            "passage about leave", [{"path": "hr/leave.pdf", "source": "rag"}]
        ),
    )


def test_agent_calls_tool_then_answers_with_citations() -> None:
    script = [
        ModelTurn("", [ToolInvocation("1", "retrieve", {"query": "leave"})]),
        ModelTurn("Grounded answer.", []),
    ]
    orch, store = _agent(script)
    conv = store.create_conversation("u1")
    msg = orch.run_turn(
        user_id="u1", conversation_id=conv["id"], text="leave policy?", tools=[_rag_tool()]
    )
    assert msg["content"] == "Grounded answer."
    assert msg["tools_used"] == ["retrieve"]
    assert msg["citations"][0]["path"] == "hr/leave.pdf"
    assert [m["role"] for m in store.list_messages(conv["id"])] == ["user", "assistant"]


def test_agent_without_tools_answers_directly() -> None:
    orch, store = _agent([ModelTurn("Direct answer.", [])])
    conv = store.create_conversation("u1")
    msg = orch.run_turn(user_id="u1", conversation_id=conv["id"], text="hi", tools=[])
    assert msg["content"] == "Direct answer."
    assert msg["tools_used"] == []


def test_agent_respects_iteration_limit() -> None:
    # Model always asks for a tool -> loop hits the cap and returns a fallback.
    script = [ModelTurn("", [ToolInvocation("1", "retrieve", {"query": "x"})])]
    orch, store = _agent(script, max_iterations=2)
    conv = store.create_conversation("u1")
    msg = orch.run_turn(user_id="u1", conversation_id=conv["id"], text="loop", tools=[_rag_tool()])
    assert "limit" in msg["content"]


def test_agent_handles_unknown_tool_call() -> None:
    script = [ModelTurn("", [ToolInvocation("1", "nope", {})]), ModelTurn("done", [])]
    orch, store = _agent(script)
    conv = store.create_conversation("u1")
    msg = orch.run_turn(user_id="u1", conversation_id=conv["id"], text="x", tools=[_rag_tool()])
    assert msg["content"] == "done"
    assert msg["citations"] == []
