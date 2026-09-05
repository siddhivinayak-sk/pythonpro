"""Tool-calling agent: RAG retrieval + MCP tools grounded into the chat, with citations.

A small, provider-agnostic loop drives an ``AgentModel`` (which the real path implements over a LangChain
model's native tool-calling). The model is injectable, so tests script tool calls + a final answer with a
fake model and fake tools. Streaming-with-tools and a full LangGraph multi-agent runtime are future work.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from ai_agent_core import apply_callbacks, get_logger

from .chat import _append_context, _content_to_text, build_model_params
from .config import ChatSettings
from .moderation import AllowAllModerationProvider, ModerationProvider
from .store import ChatStore
from .tools import AgentTool, to_openai_spec

log = get_logger("chat-agent")


@dataclass
class ToolInvocation:
    id: str
    name: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelTurn:
    content: str
    tool_calls: list[ToolInvocation] = field(default_factory=list)


class AgentModel(Protocol):
    def bind_tools(self, specs: list[dict]) -> AgentModel: ...
    def invoke(self, messages: list) -> ModelTurn: ...


class LangChainAgentModel:
    """Adapt a LangChain chat model's native tool-calling to the ``AgentModel`` interface."""

    def __init__(self, model: Any) -> None:
        self._model = model

    def bind_tools(self, specs: list[dict]) -> AgentModel:
        return LangChainAgentModel(self._model.bind_tools(specs)) if specs else self

    def invoke(self, messages: list) -> ModelTurn:
        ai = self._model.invoke(messages)
        calls = [
            ToolInvocation(id=tc.get("id", ""), name=tc.get("name", ""), args=tc.get("args") or {})
            for tc in (getattr(ai, "tool_calls", None) or [])
        ]
        return ModelTurn(content=_content_to_text(getattr(ai, "content", "")), tool_calls=calls)


AgentModelProvider = Callable[[str | None, str | None, dict[str, Any]], AgentModel]


class AgentOrchestrator:
    def __init__(
        self,
        registry: Any,
        store: ChatStore,
        settings: ChatSettings,
        *,
        model_provider: AgentModelProvider | None = None,
        moderation: ModerationProvider | None = None,
        tracing_callbacks: list[Any] | None = None,
        max_iterations: int = 4,
    ) -> None:
        self.registry = registry
        self.store = store
        self.settings = settings
        self.max_iterations = max_iterations
        self._model_provider = model_provider or self._default_provider
        self.moderation = moderation or AllowAllModerationProvider()
        self._tracing_callbacks = tracing_callbacks or []

    def _default_provider(self, connection_id, model_name, params) -> AgentModel:
        model = self.registry.get_chat_model(connection_id, model_name, **params)
        return LangChainAgentModel(apply_callbacks(model, self._tracing_callbacks))

    def _base_messages(self, conversation_id, text, images, eff) -> list:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

        messages: list = []
        if eff.get("system_prompt"):
            messages.append(SystemMessage(content=eff["system_prompt"]))
        _append_context(
            messages,
            self.store.list_context(conversation_id),
            self.settings.context_char_cap,
            SystemMessage,
        )
        for msg in self.store.recent_messages(conversation_id, int(eff.get("memory_window") or 0)):
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        if images:
            content: list[Any] = [{"type": "text", "text": text}]
            content.extend({"type": "image_url", "image_url": {"url": u}} for u in images)
            messages.append(HumanMessage(content=content))
        else:
            messages.append(HumanMessage(content=text))
        return messages

    def run_turn(
        self,
        *,
        user_id: str,
        conversation_id: str,
        text: str,
        tools: list[AgentTool],
        images: list[str] | None = None,
        connection_id: str | None = None,
        model_name: str | None = None,
        temperature: float | None = None,
    ) -> dict[str, Any]:
        from langchain_core.messages import AIMessage, ToolMessage

        eff = self.store.effective_settings(user_id, conversation_id)
        conn = connection_id or eff.get("connection_id")
        model = model_name or eff.get("model_name")
        params = build_model_params(eff, temperature=temperature)

        messages = self._base_messages(conversation_id, text, images, eff)
        self.store.add_message(conversation_id, "user", text)

        input_verdict = self.moderation.check(text, kind="input")
        if input_verdict.flagged:
            log.info("moderation_flagged", conversation_id=conversation_id, kind="input")
            return self.store.add_message(
                conversation_id,
                "assistant",
                self.settings.moderation_message,
                connection_id=conn,
                model_name=model,
                meta={
                    "moderation": {
                        "flagged": True,
                        "kind": "input",
                        "categories": input_verdict.categories,
                    }
                },
            )

        tool_map = {t.name: t for t in tools}
        agent_model = self._model_provider(conn, model, params)
        bound = agent_model.bind_tools([to_openai_spec(t) for t in tools]) if tools else agent_model

        citations: list[dict[str, Any]] = []
        tools_used: list[str] = []
        answer = ""
        for _ in range(self.max_iterations):
            turn = bound.invoke(messages)
            if not turn.tool_calls:
                answer = turn.content
                break
            messages.append(
                AIMessage(
                    content=turn.content or "",
                    tool_calls=[
                        {"name": c.name, "args": c.args, "id": c.id, "type": "tool_call"}
                        for c in turn.tool_calls
                    ],
                )
            )
            for call in turn.tool_calls:
                tool = tool_map.get(call.name)
                if tool is None:
                    messages.append(
                        ToolMessage(content=f"unknown tool: {call.name}", tool_call_id=call.id)
                    )
                    continue
                result = tool.run(**call.args)
                tools_used.append(call.name)
                citations.extend(result.citations)
                messages.append(ToolMessage(content=result.text, tool_call_id=call.id))
        else:
            answer = answer or "(reached tool-iteration limit)"

        output_verdict = self.moderation.check(answer, kind="output")
        if output_verdict.flagged:
            log.info("moderation_flagged", conversation_id=conversation_id, kind="output")
            return self.store.add_message(
                conversation_id,
                "assistant",
                self.settings.moderation_message,
                connection_id=conn,
                model_name=model,
                meta={
                    "moderation": {
                        "flagged": True,
                        "kind": "output",
                        "categories": output_verdict.categories,
                    }
                },
            )

        meta = {"citations": citations, "tools_used": sorted(set(tools_used))}
        log.info(
            "agent_turn",
            conversation_id=conversation_id,
            tools=meta["tools_used"],
            citations=len(citations),
        )
        return self.store.add_message(
            conversation_id, "assistant", answer, connection_id=conn, model_name=model, meta=meta
        )
