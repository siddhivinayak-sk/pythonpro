"""Shared metadata-filter matching for backends that filter in Python (memory store, sparse BM25).

Supports a small, backend-agnostic DSL:
- ``{field: value}``            -> equality
- ``{field: {"$eq": value}}``  -> equality
- ``{field: {"$contains": s}}``-> case-insensitive substring on the string form of the value
"""

from __future__ import annotations

from typing import Any


def matches(metadata: dict[str, Any], filters: dict[str, Any] | None) -> bool:
    if not filters:
        return True
    for key, condition in filters.items():
        value = metadata.get(key)
        if isinstance(condition, dict):
            if "$contains" in condition:
                needle = str(condition["$contains"]).lower()
                if value is None or needle not in str(value).lower():
                    return False
            if "$eq" in condition and value != condition["$eq"]:
                return False
        elif value != condition:
            return False
    return True
