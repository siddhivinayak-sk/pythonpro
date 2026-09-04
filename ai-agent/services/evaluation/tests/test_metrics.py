"""Tests for retrieval metrics (hand-verified)."""

from __future__ import annotations

import math

from ai_agent_eval.metrics import mean, ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank


def test_precision_at_k() -> None:
    assert precision_at_k(["a", "b", "c"], {"a", "c"}, 2) == 0.5  # top2 = a,b -> 1 hit / 2
    assert precision_at_k([], {"a"}, 3) == 0.0


def test_recall_at_k() -> None:
    assert recall_at_k(["a", "b"], {"a", "c"}, 5) == 0.5  # found a of {a,c}
    assert recall_at_k(["a"], set(), 5) == 0.0


def test_reciprocal_rank() -> None:
    assert reciprocal_rank(["x", "a"], {"a"}) == 0.5
    assert reciprocal_rank(["a"], {"a"}) == 1.0
    assert reciprocal_rank(["x"], {"a"}) == 0.0


def test_ndcg_at_k() -> None:
    assert ndcg_at_k(["a", "b"], {"a"}, 2) == 1.0  # relevant first -> perfect
    assert abs(ndcg_at_k(["b", "a"], {"a"}, 2) - 1.0 / math.log2(3)) < 1e-9


def test_mean() -> None:
    assert mean([1.0, 2.0, 3.0]) == 2.0
    assert mean([]) == 0.0
