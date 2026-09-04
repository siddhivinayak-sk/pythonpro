"""Retrieval metrics (pure functions, no LLM).

All operate on an ordered list of retrieved document ids vs a set of relevant ids.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence


def precision_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top = retrieved[:k]
    if not top:
        return 0.0
    relevant_set = set(relevant)
    hits = sum(1 for doc in top if doc in relevant_set)
    return hits / len(top)


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    top = set(retrieved[:k])
    return len(top & relevant_set) / len(relevant_set)


def reciprocal_rank(retrieved: Sequence[str], relevant: Iterable[str]) -> float:
    relevant_set = set(relevant)
    for index, doc in enumerate(retrieved, start=1):
        if doc in relevant_set:
            return 1.0 / index
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float:
    """Binary-relevance NDCG@k."""
    relevant_set = set(relevant)
    if not relevant_set or k <= 0:
        return 0.0
    dcg = 0.0
    for index, doc in enumerate(retrieved[:k], start=1):
        if doc in relevant_set:
            dcg += 1.0 / math.log2(index + 1)
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0
