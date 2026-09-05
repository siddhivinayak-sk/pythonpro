"""Tests for the chat orchestrator, using a fake model (no provider SDK / network)."""

from __future__ import annotations

from ai_agent_chat.chat import ChatOrchestrator
from ai_agent_chat.config import ChatSettings
from ai_agent_chat.db import Database
from ai_agent_chat.llm import build_registry
from ai_agent_chat.store import ChatStore


class _Chunk:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeModel:
    def __init__(self, recorder: list) -> None:
        self._recorder = recorder

    def invoke(self, messages: list):
        self._recorder.append(messages)
        return _Chunk("assistant reply")

    def stream(self, messages: list):
        self._recorder.append(messages)
        for piece in ["as", "sistant", " reply"]:
            yield _Chunk(piece)


def _orchestrator(recorder: list, memory_window: int = 5):
    settings = ChatSettings(db_path=":memory:", default_memory_window=memory_window)
    db = Database(settings)
    store = ChatStore(db, settings)
    registry = build_registry(settings)

    def provider(connection_id, model_name, params):
        return _FakeModel(recorder)

    return ChatOrchestrator(registry, store, settings, model_provider=provider), store


def test_complete_persists_user_and_assistant() -> None:
    recorder: list = []
    orch, store = _orchestrator(recorder)
    conv = store.create_conversation("u1")
    msg = orch.complete(
        user_id="u1",
        conversation_id=conv["id"],
        text="hello",
        connection_id="openai-prod",
        model_name="gpt-4o",
    )
    assert msg["role"] == "assistant"
    assert msg["content"] == "assistant reply"
    assert msg["connection_id"] == "openai-prod"
    assert msg["model_name"] == "gpt-4o"
    assert [m["role"] for m in store.list_messages(conv["id"])] == ["user", "assistant"]


def test_memory_window_limits_history_sent_to_model() -> None:
    recorder: list = []
    orch, store = _orchestrator(recorder, memory_window=2)
    conv = store.create_conversation("u1")
    orch.complete(user_id="u1", conversation_id=conv["id"], text="t1")
    orch.complete(user_id="u1", conversation_id=conv["id"], text="t2")
    orch.complete(user_id="u1", conversation_id=conv["id"], text="t3")
    # 3rd turn: at most 2 prior messages + the current human message.
    assert len(recorder[-1]) <= 3


def test_stream_yields_tokens_and_persists() -> None:
    recorder: list = []
    orch, store = _orchestrator(recorder)
    conv = store.create_conversation("u1")
    pieces = list(orch.stream(user_id="u1", conversation_id=conv["id"], text="hi"))
    assert "".join(pieces) == "assistant reply"
    assert store.list_messages(conv["id"])[-1]["content"] == "assistant reply"


def test_vision_images_included_in_current_turn() -> None:
    recorder: list = []
    orch, store = _orchestrator(recorder)
    conv = store.create_conversation("u1")
    orch.complete(
        user_id="u1",
        conversation_id=conv["id"],
        text="what is this?",
        images=["data:image/png;base64,AAAA"],
    )
    last_human = recorder[-1][-1]  # the current turn's HumanMessage
    assert isinstance(last_human.content, list)
    assert any(part.get("type") == "image_url" for part in last_human.content)


def test_context_files_injected_into_prompt() -> None:
    recorder: list = []
    orch, store = _orchestrator(recorder)
    conv = store.create_conversation("u1")
    store.add_context(conv["id"], "u1", "policy.txt", "PARENTAL_LEAVE_DETAILS_MARKER")
    orch.complete(user_id="u1", conversation_id=conv["id"], text="summarize the policy")
    sent = recorder[-1]
    joined = " ".join(str(getattr(m, "content", m)) for m in sent)
    assert "PARENTAL_LEAVE_DETAILS_MARKER" in joined


def test_build_model_params_maps_generation_settings() -> None:
    from ai_agent_chat.chat import build_model_params

    eff = {
        "temperature": 0.3,
        "top_p": 0.9,
        "max_tokens": 100,
        "frequency_penalty": 0.5,
        "presence_penalty": 0.2,
        "stop": "END",
    }
    assert build_model_params(eff) == {
        "temperature": 0.3,
        "top_p": 0.9,
        "max_tokens": 100,
        "frequency_penalty": 0.5,
        "presence_penalty": 0.2,
        "stop": ["END"],  # a single stop string becomes a list
    }


def test_build_model_params_omits_unset_and_blank_stop() -> None:
    from ai_agent_chat.chat import build_model_params

    assert build_model_params({"temperature": None, "stop": "   "}) == {}


def test_build_model_params_temperature_override_wins() -> None:
    from ai_agent_chat.chat import build_model_params

    assert build_model_params({"temperature": 0.7}, temperature=0.1)["temperature"] == 0.1


def test_generation_settings_flow_from_conversation_to_model() -> None:
    settings = ChatSettings(db_path=":memory:")
    store = ChatStore(Database(settings), settings)
    registry = build_registry(settings)
    seen: list[dict] = []

    class _M:
        def invoke(self, messages):
            return _Chunk("ok")

        def stream(self, messages):
            yield _Chunk("ok")

    def provider(c, m, p):
        seen.append(p)
        return _M()

    orch = ChatOrchestrator(registry, store, settings, model_provider=provider)
    conv = store.create_conversation("u1")
    store.put_setting("conversation", conv["id"], {"top_p": 0.9, "max_tokens": 50, "stop": "END"})
    orch.complete(user_id="u1", conversation_id=conv["id"], text="hi")
    assert seen[-1]["top_p"] == 0.9
    assert seen[-1]["max_tokens"] == 50
    assert seen[-1]["stop"] == ["END"]
