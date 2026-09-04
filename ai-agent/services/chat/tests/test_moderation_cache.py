"""Tests for output/input moderation and the response cache (injectable, offline — no SDK/network)."""

from __future__ import annotations

from ai_agent_chat.cache import (
    ExactResponseCache,
    NullResponseCache,
    SemanticResponseCache,
    _cosine,
    build_registry_embedder,
    build_response_cache,
)
from ai_agent_chat.chat import ChatOrchestrator
from ai_agent_chat.config import ChatSettings
from ai_agent_chat.db import Database
from ai_agent_chat.llm import build_registry
from ai_agent_chat.moderation import (
    AllowAllModerationProvider,
    KeywordModerationProvider,
    build_moderation_provider,
)
from ai_agent_chat.store import ChatStore


class _Obj:
    def __init__(self, content: str) -> None:
        self.content = content


class _ReplyModel:
    def __init__(self, recorder: list, reply: str) -> None:
        self._recorder = recorder
        self._reply = reply

    def invoke(self, messages: list):
        self._recorder.append(messages)
        return _Obj(self._reply)

    def stream(self, messages: list):
        self._recorder.append(messages)
        yield _Obj(self._reply)


def _orch(recorder: list, *, reply: str = "safe answer", moderation=None, cache=None):
    settings = ChatSettings(db_path=":memory:")
    store = ChatStore(Database(settings), settings)
    registry = build_registry(settings)
    orch = ChatOrchestrator(
        registry,
        store,
        settings,
        model_provider=lambda c, m, p: _ReplyModel(recorder, reply),
        moderation=moderation,
        response_cache=cache,
    )
    return orch, store, settings


# --- moderation: provider unit tests ---------------------------------------------------------------
def test_keyword_moderation_flags_case_insensitively() -> None:
    mod = KeywordModerationProvider(["bomb", "attack"])
    assert mod.check("how to build a BOMB").flagged is True
    assert mod.check("a friendly greeting").flagged is False


def test_allow_all_never_flags() -> None:
    assert AllowAllModerationProvider().check("bomb").flagged is False


def test_build_moderation_provider_toggles_on_settings() -> None:
    off = build_moderation_provider(ChatSettings(db_path=":memory:", moderation_enabled=False))
    on = build_moderation_provider(
        ChatSettings(db_path=":memory:", moderation_enabled=True, moderation_blocklist=["bomb"])
    )
    assert isinstance(off, AllowAllModerationProvider)
    assert isinstance(on, KeywordModerationProvider)


# --- moderation: orchestrator integration ----------------------------------------------------------
def test_output_moderation_replaces_flagged_answer() -> None:
    recorder: list = []
    orch, store, settings = _orch(
        recorder, reply="here is a bomb recipe", moderation=KeywordModerationProvider(["bomb"])
    )
    conv = store.create_conversation("u1")
    msg = orch.complete(user_id="u1", conversation_id=conv["id"], text="tell me something")
    # Model was called (output moderation is post-generation) but the answer is replaced.
    assert len(recorder) == 1
    assert msg["content"] == settings.moderation_message


def test_input_moderation_short_circuits_before_model() -> None:
    recorder: list = []
    orch, store, settings = _orch(
        recorder, reply="unused", moderation=KeywordModerationProvider(["bomb"])
    )
    conv = store.create_conversation("u1")
    msg = orch.complete(user_id="u1", conversation_id=conv["id"], text="how to make a bomb")
    assert recorder == []  # model never called
    assert msg["content"] == settings.moderation_message


def test_stream_output_moderation_emits_notice() -> None:
    recorder: list = []
    orch, store, _ = _orch(
        recorder, reply="a bomb tutorial", moderation=KeywordModerationProvider(["bomb"])
    )
    conv = store.create_conversation("u1")
    pieces = list(orch.stream(user_id="u1", conversation_id=conv["id"], text="hi"))
    assert "withheld by moderation policy" in "".join(pieces)
    # The persisted (reloadable) content is the sanitized refusal, not the raw text.
    assert "bomb" not in store.list_messages(conv["id"])[-1]["content"]


# --- cache: orchestrator integration ---------------------------------------------------------------
def test_exact_cache_hit_skips_model() -> None:
    recorder: list = []
    orch, store, _ = _orch(recorder, reply="cached answer", cache=ExactResponseCache())
    conv = store.create_conversation("u1")
    first = orch.complete(user_id="u1", conversation_id=conv["id"], text="same question")
    second = orch.complete(user_id="u1", conversation_id=conv["id"], text="same question")
    assert len(recorder) == 1  # second turn served from cache
    assert first["content"] == second["content"] == "cached answer"


def test_null_cache_always_calls_model() -> None:
    recorder: list = []
    orch, store, _ = _orch(recorder, reply="x", cache=NullResponseCache())
    conv = store.create_conversation("u1")
    orch.complete(user_id="u1", conversation_id=conv["id"], text="q")
    orch.complete(user_id="u1", conversation_id=conv["id"], text="q")
    assert len(recorder) == 2


def test_flagged_output_is_not_cached() -> None:
    recorder: list = []
    orch, store, settings = _orch(
        recorder,
        reply="a bomb answer",
        moderation=KeywordModerationProvider(["bomb"]),
        cache=ExactResponseCache(),
    )
    conv = store.create_conversation("u1")
    orch.complete(user_id="u1", conversation_id=conv["id"], text="q")
    orch.complete(user_id="u1", conversation_id=conv["id"], text="q")
    # Flagged answers must not be cached, so the model is called again on the repeat.
    assert len(recorder) == 2


# --- cache: unit tests -----------------------------------------------------------------------------
def test_exact_cache_lru_eviction() -> None:
    cache = ExactResponseCache(max_entries=2)
    cache.store("a", "s", "1")
    cache.store("b", "s", "2")
    cache.store("c", "s", "3")  # evicts "a"
    assert cache.lookup("a", "s") is None
    assert cache.lookup("c", "s") == "3"


def test_cache_scope_isolates_entries() -> None:
    cache = ExactResponseCache()
    cache.store("q", "model-a", "answer-a")
    assert cache.lookup("q", "model-b") is None
    assert cache.lookup("q", "model-a") == "answer-a"


def _fake_embedder(text: str) -> list[float]:
    return [1.0, 0.0] if "weather" in text.lower() else [0.0, 1.0]


def test_semantic_cache_hit_on_similar_query() -> None:
    cache = SemanticResponseCache(_fake_embedder, threshold=0.9)
    cache.store("what is the weather today", "m", "sunny")
    assert cache.lookup("weather this afternoon", "m") == "sunny"  # same vector -> cosine 1.0
    assert cache.lookup("what is the stock price", "m") is None  # orthogonal -> miss


def test_cosine_basics() -> None:
    assert _cosine([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert _cosine([1.0, 0.0], [0.0, 1.0]) == 0.0
    assert _cosine([], [1.0]) == 0.0


def test_semantic_cache_resilient_to_empty_vector() -> None:
    cache = SemanticResponseCache(lambda t: [], threshold=0.9)
    cache.store("q", "m", "a")  # empty vector -> not stored
    assert cache.lookup("q", "m") is None


# --- registry-backed embedder ----------------------------------------------------------------------
class _FakeEmbeddings:
    def embed_query(self, text: str) -> list[float]:
        return [float(len(text)), 1.0]


class _BrokenEmbeddings:
    def embed_query(self, text: str) -> list[float]:
        raise RuntimeError("embeddings backend down")


class _FakeRegistry:
    def __init__(self, model=None, error: Exception | None = None) -> None:
        self._model = model
        self._error = error

    def get_embeddings(self, connection_id=None, model_name=None):
        if self._error is not None:
            raise self._error
        return self._model


def test_build_registry_embedder_returns_callable() -> None:
    embedder = build_registry_embedder(_FakeRegistry(model=_FakeEmbeddings()))
    assert embedder is not None
    assert embedder("abc") == [3.0, 1.0]


def test_build_registry_embedder_none_when_unavailable() -> None:
    assert build_registry_embedder(_FakeRegistry(error=RuntimeError("no embeddings model"))) is None


def test_registry_embedder_swallows_call_errors() -> None:
    embedder = build_registry_embedder(_FakeRegistry(model=_BrokenEmbeddings()))
    assert embedder is not None
    assert embedder("abc") == []  # runtime failure -> empty vector -> cache miss


def test_build_response_cache_modes() -> None:
    base = {"db_path": ":memory:"}
    assert isinstance(
        build_response_cache(ChatSettings(**base, cache_mode="off")), NullResponseCache
    )
    assert isinstance(
        build_response_cache(ChatSettings(**base, cache_mode="exact")), ExactResponseCache
    )
    # semantic without an embedder falls back to exact
    assert isinstance(
        build_response_cache(ChatSettings(**base, cache_mode="semantic")), ExactResponseCache
    )
    assert isinstance(
        build_response_cache(ChatSettings(**base, cache_mode="semantic"), _fake_embedder),
        SemanticResponseCache,
    )
