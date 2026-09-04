"""Lightweight application database (SQLAlchemy 2.0).

Engine-portable schema so the store works on **SQLite** (default) or **DuckDB**, with a configurable file
location (``:memory:`` supported for tests via a shared static pool). Holds users, sessions, conversations,
messages, attachments, and settings.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from sqlalchemy import (
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker
from sqlalchemy.pool import StaticPool

from .config import ChatSettings


def _uuid() -> str:
    return uuid4().hex


def _now() -> datetime:
    # Naive UTC: SQLite/DuckDB round-trip datetimes without tzinfo, so we keep everything naive-UTC
    # to avoid aware/naive comparison errors when checking session expiry.
    return datetime.now(UTC).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    subject: Mapped[str] = mapped_column(
        String(255), unique=True, index=True
    )  # username (local) or sub (oidc)
    source: Mapped[str] = mapped_column(String(16), default="local")  # local | oidc
    email: Mapped[str | None] = mapped_column(String(255), nullable=True)
    display_name: Mapped[str | None] = mapped_column(String(255), nullable=True)
    role: Mapped[str] = mapped_column(String(32), default="user")  # admin | user
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)


class Session(Base):
    __tablename__ = "sessions"
    token: Mapped[str] = mapped_column(String(128), primary_key=True)
    user_id: Mapped[str] = mapped_column(String(64), ForeignKey("users.id"), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    expires_at: Mapped[datetime] = mapped_column(DateTime)


class Conversation(Base):
    __tablename__ = "conversations"
    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(String(64), ForeignKey("users.id"), index=True)
    title: Mapped[str] = mapped_column(String(255), default="New chat")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_now, onupdate=_now)
    archived: Mapped[bool] = mapped_column(Boolean, default=False)
    pinned: Mapped[bool] = mapped_column(Boolean, default=False)


class Message(Base):
    __tablename__ = "messages"
    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    conversation_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("conversations.id"), index=True
    )
    role: Mapped[str] = mapped_column(String(16))  # user | assistant | system | tool
    content: Mapped[str] = mapped_column(Text, default="")
    connection_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    model_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    tokens_in: Mapped[int] = mapped_column(Integer, default=0)
    tokens_out: Mapped[int] = mapped_column(Integer, default=0)
    meta: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON: citations, tools_used
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)


class Workflow(Base):
    __tablename__ = "workflows"
    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(String(64), ForeignKey("users.id"), index=True)
    name: Mapped[str] = mapped_column(String(255), default="Workflow")
    definition: Mapped[str] = mapped_column(Text, default="{}")  # JSON: {steps: [...]}
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_now, onupdate=_now)


class WorkflowRun(Base):
    __tablename__ = "workflow_runs"
    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    workflow_id: Mapped[str] = mapped_column(String(64), ForeignKey("workflows.id"), index=True)
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    status: Mapped[str] = mapped_column(String(16), default="completed")  # completed | failed
    inputs: Mapped[str] = mapped_column(Text, default="{}")  # JSON
    output: Mapped[str] = mapped_column(Text, default="")
    trace: Mapped[str] = mapped_column(Text, default="[]")  # JSON: per-step results
    started_at: Mapped[datetime] = mapped_column(DateTime, default=_now)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class ContextFile(Base):
    """A text document attached to a conversation as ad-hoc grounding context."""

    __tablename__ = "context_files"
    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    conversation_id: Mapped[str] = mapped_column(
        String(64), ForeignKey("conversations.id"), index=True
    )
    user_id: Mapped[str] = mapped_column(String(64), index=True)
    filename: Mapped[str] = mapped_column(String(255))
    text: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)


class Attachment(Base):
    __tablename__ = "attachments"
    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    user_id: Mapped[str] = mapped_column(String(64), ForeignKey("users.id"), index=True)
    message_id: Mapped[str | None] = mapped_column(
        String(64), ForeignKey("messages.id"), nullable=True
    )
    kind: Mapped[str] = mapped_column(String(16), default="image")
    path: Mapped[str] = mapped_column(String(1024))
    mime: Mapped[str] = mapped_column(String(128), default="application/octet-stream")
    size: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_now)


class Setting(Base):
    __tablename__ = "settings"
    id: Mapped[str] = mapped_column(String(64), primary_key=True, default=_uuid)
    scope: Mapped[str] = mapped_column(String(16), index=True)  # user | conversation
    owner_id: Mapped[str] = mapped_column(String(64), index=True)
    data: Mapped[str] = mapped_column(Text, default="{}")  # JSON blob
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=_now, onupdate=_now)


def _engine_url(settings: ChatSettings) -> str:
    if settings.db_path == ":memory:":
        return "sqlite://" if settings.db_engine == "sqlite" else "duckdb:///:memory:"
    if settings.db_engine == "sqlite":
        return f"sqlite:///{settings.db_path}"
    if settings.db_engine == "duckdb":
        return f"duckdb:///{settings.db_path}"
    raise ValueError(f"unsupported db_engine '{settings.db_engine}' (use sqlite or duckdb)")


class Database:
    def __init__(self, settings: ChatSettings) -> None:
        url = _engine_url(settings)
        if settings.db_path != ":memory:" and settings.db_engine == "sqlite":
            parent = Path(settings.db_path).parent
            if str(parent) not in ("", "."):
                parent.mkdir(parents=True, exist_ok=True)

        if url == "sqlite://":  # shared in-memory DB across threads (tests)
            self.engine = create_engine(
                url, connect_args={"check_same_thread": False}, poolclass=StaticPool
            )
        elif url.startswith("sqlite:"):
            self.engine = create_engine(url, connect_args={"check_same_thread": False})
        else:
            self.engine = create_engine(url)

        Base.metadata.create_all(self.engine)
        self.session_factory = sessionmaker(self.engine, expire_on_commit=False)

    def session(self):
        return self.session_factory()
