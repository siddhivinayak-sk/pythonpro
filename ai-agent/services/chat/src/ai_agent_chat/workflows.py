"""A small workflow engine: run a saved multi-step automation over a shared context.

Step types: ``llm`` (prompt → model), ``rag`` (retrieve from a collection), ``tool`` (call an agent tool).
Strings support ``{{inputs.x}}`` and ``{{step_id}}`` templating from prior outputs. Model, RAG client, and
tools are injected, so the engine is fully testable with fakes.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from ai_agent_core import get_logger

from .chat import _content_to_text
from .rag_client import RetrievalClient
from .tools import AgentTool

log = get_logger("workflow")

_VAR = re.compile(r"\{\{\s*([\w.]+)\s*\}\}")

ModelProvider = Callable[[str | None, str | None, dict[str, Any]], Any]


class WorkflowError(Exception):
    pass


class WorkflowEngine:
    def __init__(
        self,
        *,
        model_provider: ModelProvider,
        rag_client: RetrievalClient | None = None,
        tools: dict[str, AgentTool] | None = None,
    ) -> None:
        self._model_provider = model_provider
        self._rag_client = rag_client
        self._tools = tools or {}

    def _template(self, text: str, context: dict[str, Any]) -> str:
        def repl(match: re.Match) -> str:
            value: Any = context
            for part in match.group(1).split("."):
                value = value.get(part, "") if isinstance(value, dict) else ""
            return str(value)

        return _VAR.sub(repl, text)

    def _run_step(self, step: dict[str, Any], context: dict[str, Any]) -> str:
        step_type = step.get("type")
        if step_type == "llm":
            from langchain_core.messages import HumanMessage, SystemMessage

            messages: list = []
            if step.get("system"):
                messages.append(SystemMessage(content=self._template(step["system"], context)))
            messages.append(HumanMessage(content=self._template(step.get("prompt", ""), context)))
            model = self._model_provider(step.get("connection_id"), step.get("model_name"), {})
            return _content_to_text(getattr(model.invoke(messages), "content", ""))

        if step_type == "rag":
            if self._rag_client is None:
                raise WorkflowError("rag step requires a configured RAG client")
            query = self._template(step.get("query", ""), context)
            hits = self._rag_client.retrieve(step["collection"], query, step.get("k", 5))
            return "\n\n".join(h.get("text", "") for h in hits)

        if step_type == "tool":
            tool = self._tools.get(step.get("tool", ""))
            if tool is None:
                raise WorkflowError(f"unknown tool: {step.get('tool')}")
            args = {
                key: self._template(val, context) if isinstance(val, str) else val
                for key, val in (step.get("args") or {}).items()
            }
            return tool.run(**args).text

        raise WorkflowError(f"unknown step type: {step_type}")

    def run(self, definition: dict[str, Any], inputs: dict[str, Any]) -> dict[str, Any]:
        context: dict[str, Any] = {"inputs": inputs}
        trace: list[dict[str, Any]] = []
        last_output = ""
        status = "completed"

        for step in definition.get("steps", []):
            step_id = step.get("id", f"step{len(trace)}")
            try:
                output = self._run_step(step, context)
            except Exception as exc:  # noqa: BLE001 - record and stop the run
                trace.append({"id": step_id, "type": step.get("type"), "error": str(exc)})
                status = "failed"
                log.warning("workflow_step_failed", step=step_id, error=str(exc))
                break
            context[step_id] = output
            last_output = output
            trace.append({"id": step_id, "type": step.get("type"), "output": output})

        output_tmpl = definition.get("output")
        final = self._template(output_tmpl, context) if output_tmpl else last_output
        return {"status": status, "output": final, "trace": trace}
