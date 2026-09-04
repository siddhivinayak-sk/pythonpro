"""Tests for the TTL cache (deterministic via an injected clock)."""

from __future__ import annotations

from ai_agent_mcp_web_search.cache import TTLCache


def test_get_set_returns_value_before_expiry() -> None:
    clock = {"t": 0.0}
    cache = TTLCache(ttl_seconds=10, time_func=lambda: clock["t"])
    cache.set("k", {"v": 1})
    assert cache.get("k") == {"v": 1}


def test_entry_expires_after_ttl() -> None:
    clock = {"t": 0.0}
    cache = TTLCache(ttl_seconds=10, time_func=lambda: clock["t"])
    cache.set("k", "v")
    clock["t"] = 10.1
    assert cache.get("k") is None


def test_missing_key_returns_none() -> None:
    assert TTLCache().get("absent") is None


def test_lru_eviction_when_full() -> None:
    cache = TTLCache(ttl_seconds=1000, max_entries=2, time_func=lambda: 0.0)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.get("a")  # touch 'a' so 'b' becomes least-recently-used
    cache.set("c", 3)  # evicts 'b'
    assert cache.get("a") == 1
    assert cache.get("c") == 3
    assert cache.get("b") is None
    assert len(cache) == 2
