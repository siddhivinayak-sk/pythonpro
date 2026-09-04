"""A small thread-safe TTL cache for search results.

Kept dependency-free. The clock is injectable so tests can advance time deterministically.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Hashable
from typing import Any


class TTLCache:
    def __init__(
        self,
        ttl_seconds: float = 300.0,
        max_entries: int = 1000,
        time_func: Callable[[], float] = time.monotonic,
    ) -> None:
        self.ttl = ttl_seconds
        self.max_entries = max_entries
        self._time = time_func
        self._store: OrderedDict[Hashable, tuple[float, Any]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: Hashable) -> Any | None:
        with self._lock:
            item = self._store.get(key)
            if item is None:
                return None
            expiry, value = item
            if expiry <= self._time():
                self._store.pop(key, None)
                return None
            self._store.move_to_end(key)  # recently used
            return value

    def set(self, key: Hashable, value: Any) -> None:
        with self._lock:
            if key not in self._store and len(self._store) >= self.max_entries:
                self._store.popitem(last=False)  # evict least-recently-used
            self._store[key] = (self._time() + self.ttl, value)
            self._store.move_to_end(key)

    def __len__(self) -> int:
        return len(self._store)
