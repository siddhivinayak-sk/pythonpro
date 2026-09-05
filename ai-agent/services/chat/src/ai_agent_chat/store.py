"""Persistence for conversations, messages, and settings (over the app DB).

Returns plain dicts (JSON-serialisable) to keep the API layer simple and avoid detached-ORM pitfalls.
Settings precedence: system defaults < user settings < conversation settings (< per-request overrides,
applied by the chat orchestrator).
"""

from __future__ import annotations

import json
from typing import Any

from sqlalchemy import delete, select

from .config import ChatSettings
from .db import (
    ContextFile,
    Conversation,
    Database,
    Message,
    Setting,
    Workflow,
    WorkflowRun,
    _now,
)


def _conv_dict(c: Conversation) -> dict[str, Any]:
    return {
        "id": c.id,
        "title": c.title,
        "created_at": c.created_at.isoformat(),
        "updated_at": c.updated_at.isoformat(),
        "archived": c.archived,
        "pinned": c.pinned,
    }


def _msg_dict(m: Message) -> dict[str, Any]:
    meta = json.loads(m.meta) if m.meta else {}
    return {
        "id": m.id,
        "role": m.role,
        "content": m.content,
        "connection_id": m.connection_id,
        "model_name": m.model_name,
        "tokens_in": m.tokens_in,
        "tokens_out": m.tokens_out,
        "citations": meta.get("citations", []),
        "tools_used": meta.get("tools_used", []),
        "created_at": m.created_at.isoformat(),
    }


def _wf_dict(w: Workflow) -> dict[str, Any]:
    return {
        "id": w.id,
        "name": w.name,
        "definition": json.loads(w.definition),
        "created_at": w.created_at.isoformat(),
        "updated_at": w.updated_at.isoformat(),
    }


def _run_dict(r: WorkflowRun) -> dict[str, Any]:
    return {
        "id": r.id,
        "workflow_id": r.workflow_id,
        "status": r.status,
        "inputs": json.loads(r.inputs),
        "output": r.output,
        "trace": json.loads(r.trace),
        "started_at": r.started_at.isoformat(),
        "finished_at": r.finished_at.isoformat() if r.finished_at else None,
    }


class ChatStore:
    def __init__(self, db: Database, settings: ChatSettings) -> None:
        self.db = db
        self.settings = settings

    # -- conversations --
    def create_conversation(self, user_id: str, title: str = "New chat") -> dict[str, Any]:
        with self.db.session() as s:
            conv = Conversation(user_id=user_id, title=title)
            s.add(conv)
            s.commit()
            return _conv_dict(conv)

    def list_conversations(self, user_id: str) -> list[dict[str, Any]]:
        with self.db.session() as s:
            rows = s.scalars(
                select(Conversation)
                .where(Conversation.user_id == user_id)
                .order_by(Conversation.pinned.desc(), Conversation.updated_at.desc())
            ).all()
            return [_conv_dict(c) for c in rows]

    def get_conversation(self, user_id: str, conversation_id: str) -> dict[str, Any] | None:
        with self.db.session() as s:
            conv = s.get(Conversation, conversation_id)
            if conv is None or conv.user_id != user_id:
                return None
            return _conv_dict(conv)

    def rename_conversation(self, user_id: str, conversation_id: str, title: str) -> bool:
        with self.db.session() as s:
            conv = s.get(Conversation, conversation_id)
            if conv is None or conv.user_id != user_id:
                return False
            conv.title = title
            s.commit()
            return True

    def delete_conversation(self, user_id: str, conversation_id: str) -> bool:
        with self.db.session() as s:
            conv = s.get(Conversation, conversation_id)
            if conv is None or conv.user_id != user_id:
                return False
            s.execute(delete(Message).where(Message.conversation_id == conversation_id))
            s.execute(delete(ContextFile).where(ContextFile.conversation_id == conversation_id))
            s.delete(conv)
            s.commit()
            return True

    # -- messages --
    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        *,
        connection_id: str | None = None,
        model_name: str | None = None,
        tokens_in: int = 0,
        tokens_out: int = 0,
        meta: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with self.db.session() as s:
            msg = Message(
                conversation_id=conversation_id,
                role=role,
                content=content,
                connection_id=connection_id,
                model_name=model_name,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                meta=json.dumps(meta) if meta else None,
            )
            s.add(msg)
            conv = s.get(Conversation, conversation_id)
            if conv is not None:
                conv.updated_at = _now()  # bump activity for history ordering
                # Auto-title from the first user message.
                if role == "user" and conv.title == "New chat":
                    conv.title = (
                        (content[:60] + "…") if len(content) > 60 else (content or "New chat")
                    )
            s.commit()
            return _msg_dict(msg)

    def list_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        with self.db.session() as s:
            rows = s.scalars(
                select(Message)
                .where(Message.conversation_id == conversation_id)
                .order_by(Message.created_at.asc(), Message.id.asc())
            ).all()
            return [_msg_dict(m) for m in rows]

    def recent_messages(self, conversation_id: str, limit: int) -> list[dict[str, Any]]:
        messages = self.list_messages(conversation_id)
        return messages[-limit:] if limit > 0 else messages

    # -- settings --
    def _system_defaults(self) -> dict[str, Any]:
        return {
            "temperature": self.settings.default_temperature,
            "memory_window": self.settings.default_memory_window,
            "theme": self.settings.default_theme,
            "system_prompt": None,
            "connection_id": None,
            "model_name": None,
            "rag_collections": [],
            "mcp_servers": [],
        }

    def get_setting(self, scope: str, owner_id: str) -> dict[str, Any]:
        with self.db.session() as s:
            row = s.scalar(
                select(Setting).where(Setting.scope == scope, Setting.owner_id == owner_id)
            )
            return json.loads(row.data) if row else {}

    def put_setting(self, scope: str, owner_id: str, data: dict[str, Any]) -> dict[str, Any]:
        """Merge ``data`` into the scope's stored settings (PATCH semantics).

        Merging lets partial updates (e.g. just the model from the header selector) coexist with other
        overrides for the same scope instead of clobbering them.
        """
        with self.db.session() as s:
            row = s.scalar(
                select(Setting).where(Setting.scope == scope, Setting.owner_id == owner_id)
            )
            merged = ({} if row is None or not row.data else json.loads(row.data)) | data
            if row is None:
                row = Setting(scope=scope, owner_id=owner_id, data=json.dumps(merged))
                s.add(row)
            else:
                row.data = json.dumps(merged)
            s.commit()
            return merged

    def effective_settings(
        self, user_id: str, conversation_id: str | None = None
    ) -> dict[str, Any]:
        merged = self._system_defaults()
        merged.update(self.get_setting("user", user_id))
        if conversation_id:
            merged.update(self.get_setting("conversation", conversation_id))
        return merged

    # -- workflows --
    def create_workflow(
        self, user_id: str, name: str, definition: dict[str, Any]
    ) -> dict[str, Any]:
        with self.db.session() as s:
            wf = Workflow(user_id=user_id, name=name, definition=json.dumps(definition))
            s.add(wf)
            s.commit()
            return _wf_dict(wf)

    def list_workflows(self, user_id: str) -> list[dict[str, Any]]:
        with self.db.session() as s:
            rows = s.scalars(
                select(Workflow)
                .where(Workflow.user_id == user_id)
                .order_by(Workflow.updated_at.desc())
            ).all()
            return [_wf_dict(w) for w in rows]

    def get_workflow(self, user_id: str, workflow_id: str) -> dict[str, Any] | None:
        with self.db.session() as s:
            wf = s.get(Workflow, workflow_id)
            if wf is None or wf.user_id != user_id:
                return None
            return _wf_dict(wf)

    def delete_workflow(self, user_id: str, workflow_id: str) -> bool:
        with self.db.session() as s:
            wf = s.get(Workflow, workflow_id)
            if wf is None or wf.user_id != user_id:
                return False
            s.execute(delete(WorkflowRun).where(WorkflowRun.workflow_id == workflow_id))
            s.delete(wf)
            s.commit()
            return True

    def record_run(
        self,
        workflow_id: str,
        user_id: str,
        *,
        status: str,
        inputs: dict[str, Any],
        output: str,
        trace: list[dict[str, Any]],
        started_at,
    ) -> dict[str, Any]:
        with self.db.session() as s:
            run = WorkflowRun(
                workflow_id=workflow_id,
                user_id=user_id,
                status=status,
                inputs=json.dumps(inputs),
                output=output,
                trace=json.dumps(trace),
                started_at=started_at,
                finished_at=_now(),
            )
            s.add(run)
            s.commit()
            return _run_dict(run)

    def list_runs(self, workflow_id: str) -> list[dict[str, Any]]:
        with self.db.session() as s:
            rows = s.scalars(
                select(WorkflowRun)
                .where(WorkflowRun.workflow_id == workflow_id)
                .order_by(WorkflowRun.started_at.desc())
            ).all()
            return [_run_dict(r) for r in rows]

    # -- context files --
    def add_context(
        self, conversation_id: str, user_id: str, filename: str, text: str
    ) -> dict[str, Any]:
        with self.db.session() as s:
            ctx = ContextFile(
                conversation_id=conversation_id, user_id=user_id, filename=filename, text=text
            )
            s.add(ctx)
            s.commit()
            return {"id": ctx.id, "filename": ctx.filename, "chars": len(ctx.text)}

    def list_context(self, conversation_id: str) -> list[dict[str, Any]]:
        with self.db.session() as s:
            rows = s.scalars(
                select(ContextFile)
                .where(ContextFile.conversation_id == conversation_id)
                .order_by(ContextFile.created_at.asc())
            ).all()
            return [{"id": c.id, "filename": c.filename, "text": c.text} for c in rows]

    def delete_context(self, conversation_id: str, context_id: str, user_id: str) -> bool:
        with self.db.session() as s:
            ctx = s.get(ContextFile, context_id)
            if ctx is None or ctx.user_id != user_id or ctx.conversation_id != conversation_id:
                return False
            s.delete(ctx)
            s.commit()
            return True
