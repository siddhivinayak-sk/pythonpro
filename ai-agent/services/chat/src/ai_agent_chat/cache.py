"""Response caching via a pluggable ``ResponseCache``.

Three built-ins, all injectable and offline-testable:

* ``NullResponseCache`` (default) — caching disabled.
* ``ExactResponseCache`` — in-memory exact match on the normalised query, scoped by model.
* ``SemanticResponseCache`` — returns a cached answer when a prior query in the same scope is within a
  cosine-similarity ``threshold``. The embedder is **injected**: the app auto-wires a registry-backed one
  (``build_registry_embedder``) in semantic mode, while tests use a fake — so no embedding SDK is required
  for the default install or the test suite.

Caching keys on the latest user query scoped by the model reference. This is the conventional
"semantic cache" shape (as in GPTCache); it does **not** account for divergent earlier conversation, so it
is **off by default** and best suited to stateless / FAQ-style assistants.
"""

from __future__ import annotations

import hashlib
import math
import threading
from collections import OrderedDict
from collections.abc import Callable
from typing import Protocol, runtime_checkable

from ai_agent_core import LLMConnectionRegistry, get_logger

from .config import ChatSettings

log = get_logger("cache")

Embedder = Callable[[str], list[float]]


@runtime_checkable
class ResponseCache(Protocol):
    def lookup(self, query: str, scope: str) -> str | None: ...
    def store(self, query: str, scope: str, response: str) -> None: ...


class NullResponseCache:
    """Default cache: never hits, never stores."""

    def lookup(self, query: str, scope: str) -> str | None:
        return None

    def store(self, query: str, scope: str, response: str) -> None:
        return None


class ExactResponseCache:
    """Thread-safe, bounded (LRU) exact-match cache keyed by ``(scope, normalised query)``."""

    def __init__(self, max_entries: int = 512) -> None:
        self._max = max(1, max_entries)
        self._data: OrderedDict[str, str] = OrderedDict()
        self._lock = threading.Lock()

    @staticmethod
    def _key(query: str, scope: str) -> str:
        return hashlib.sha256(f"{scope}\x00{query.strip().lower()}".encode()).hexdigest()

    def lookup(self, query: str, scope: str) -> str | None:
        key = self._key(query, scope)
        with self._lock:
            if key not in self._data:
                return None
            self._data.move_to_end(key)
            return self._data[key]

    def store(self, query: str, scope: str, response: str) -> None:
        key = self._key(query, scope)
        with self._lock:
            self._data[key] = response
            self._data.move_to_end(key)
            while len(self._data) > self._max:
                self._data.popitem(last=False)


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


class SemanticResponseCache:
    """Embedding-similarity cache. ``embedder`` maps text -> vector and is injected."""

    def __init__(self, embedder: Embedder, threshold: float = 0.95, max_entries: int = 512) -> None:
        self._embedder = embedder
        self._threshold = threshold
        self._max = max(1, max_entries)
        self._entries: dict[str, list[tuple[list[float], str]]] = {}
        self._lock = threading.Lock()

    def lookup(self, query: str, scope: str) -> str | None:
        vec = self._embedder(query)
        if not vec:  # embedder failed/unavailable -> treat as a miss (resilient)
            return None
        with self._lock:
            best_response: str | None = None
            best_sim = -1.0
            for stored_vec, response in self._entries.get(scope, []):
                sim = _cosine(vec, stored_vec)
                if sim > best_sim:
                    best_sim, best_response = sim, response
        if best_response is not None and best_sim >= self._threshold:
            log.info("cache_hit", mode="semantic", similarity=round(best_sim, 4))
            return best_response
        return None

    def store(self, query: str, scope: str, response: str) -> None:
        vec = self._embedder(query)
        if not vec:  # don't store useless empty vectors
            return
        with self._lock:
            bucket = self._entries.setdefault(scope, [])
            bucket.append((vec, response))
            while len(bucket) > self._max:
                bucket.pop(0)


def build_registry_embedder(
    registry: LLMConnectionRegistry,
    *,
    connection_id: str | None = None,
    model_name: str | None = None,
) -> Embedder | None:
    """Build an :data:`Embedder` backed by the registry's embeddings model, or ``None`` if unavailable.

    Model construction happens once, up front; if it fails (no embeddings model configured, missing SDK),
    ``None`` is returned so the caller falls back to exact matching. The returned callable also swallows
    per-call embedding errors (returning ``[]``) so a transient embeddings outage degrades to a cache miss
    rather than failing the chat request.
    """
    try:
        model = registry.get_embeddings(connection_id, model_name)
    except Exception as exc:
        log.warning("semantic_embedder_unavailable", error=str(exc))
        return None

    def _embed(text: str) -> list[float]:
        try:
            return list(model.embed_query(text))
        except Exception as exc:  # pragma: no cover - depends on live embeddings backend
            log.warning("embed_query_failed", error=str(exc))
            return []

    return _embed


def build_response_cache(settings: ChatSettings, embedder: Embedder | None = None) -> ResponseCache:
    """Construct the configured cache. Semantic mode without an embedder falls back to exact matching."""
    mode = (settings.cache_mode or "off").lower()
    if mode == "off":
        return NullResponseCache()
    if mode == "exact":
        return ExactResponseCache(settings.cache_max_entries)
    if mode == "semantic":
        if embedder is None:
            log.warning(
                "cache_semantic_without_embedder", detail="falling back to exact-match cache"
            )
            return ExactResponseCache(settings.cache_max_entries)
        return SemanticResponseCache(
            embedder, settings.cache_similarity_threshold, settings.cache_max_entries
        )
    log.warning("cache_mode_unknown", mode=mode)
    return NullResponseCache()
