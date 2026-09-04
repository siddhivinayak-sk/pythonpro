"""Tests for the conversation/message/settings store."""

from __future__ import annotations

from ai_agent_chat.config import ChatSettings
from ai_agent_chat.db import Database
from ai_agent_chat.store import ChatStore


def _store(**overrides) -> ChatStore:
    settings = ChatSettings(db_path=":memory:", **overrides)
    return ChatStore(Database(settings), settings)


def test_conversation_crud_and_ownership() -> None:
    store = _store()
    conv = store.create_conversation("u1", "Hi")
    assert conv["title"] == "Hi"
    assert any(c["id"] == conv["id"] for c in store.list_conversations("u1"))
    assert store.get_conversation("u2", conv["id"]) is None  # not owner
    assert store.rename_conversation("u1", conv["id"], "Renamed") is True
    assert store.get_conversation("u1", conv["id"])["title"] == "Renamed"
    assert store.delete_conversation("u1", conv["id"]) is True
    assert store.get_conversation("u1", conv["id"]) is None


def test_messages_and_autotitle() -> None:
    store = _store()
    conv = store.create_conversation("u1")  # default title "New chat"
    store.add_message(conv["id"], "user", "What is the parental leave policy?")
    assert store.get_conversation("u1", conv["id"])["title"].startswith("What is")
    store.add_message(conv["id"], "assistant", "It is ...")
    assert [m["role"] for m in store.list_messages(conv["id"])] == ["user", "assistant"]


def test_recent_messages_window() -> None:
    store = _store()
    conv = store.create_conversation("u1")
    for i in range(10):
        store.add_message(conv["id"], "user", f"m{i}")
    assert len(store.recent_messages(conv["id"], 3)) == 3


def test_settings_precedence() -> None:
    store = _store(default_temperature=0.7)
    assert store.effective_settings("u1")["temperature"] == 0.7
    store.put_setting("user", "u1", {"temperature": 0.1, "theme": "dark"})
    conv = store.create_conversation("u1")
    store.put_setting("conversation", conv["id"], {"temperature": 0.9})
    eff = store.effective_settings("u1", conv["id"])
    assert eff["temperature"] == 0.9  # conversation overrides user
    assert eff["theme"] == "dark"  # inherited from user


def test_context_files_crud() -> None:
    store = _store()
    conv = store.create_conversation("u1")
    added = store.add_context(conv["id"], "u1", "notes.txt", "some context text")
    assert added["chars"] == len("some context text")
    listed = store.list_context(conv["id"])
    assert listed[0]["filename"] == "notes.txt"
    assert listed[0]["text"] == "some context text"
    assert store.delete_context(conv["id"], added["id"], "u1") is True
    assert store.list_context(conv["id"]) == []
