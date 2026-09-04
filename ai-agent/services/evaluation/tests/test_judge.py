"""Tests for the heuristic judge and score parsing."""

from __future__ import annotations

import pytest
from ai_agent_eval.judge import HeuristicJudge, _parse_score


def test_faithfulness_high_when_supported() -> None:
    judge = HeuristicJudge()
    score = judge.score(
        "faithfulness",
        answer="paid parental leave",
        contexts=["employees receive paid parental leave"],
    )
    assert score == 1.0


def test_faithfulness_low_when_unsupported() -> None:
    judge = HeuristicJudge()
    score = judge.score(
        "faithfulness", answer="quantum entanglement supremacy", contexts=["paid parental leave"]
    )
    assert score < 0.5


def test_answer_relevancy() -> None:
    judge = HeuristicJudge()
    assert (
        judge.score(
            "answer_relevancy", question="parental leave policy", answer="parental leave is paid"
        )
        > 0.5
    )


def test_context_recall_full() -> None:
    judge = HeuristicJudge()
    assert (
        judge.score(
            "context_recall",
            ground_truth="paid parental leave",
            contexts=["employees get paid parental leave"],
        )
        == 1.0
    )


def test_answer_correctness() -> None:
    judge = HeuristicJudge()
    assert (
        judge.score(
            "answer_correctness", answer="paid parental leave", ground_truth="paid parental leave"
        )
        == 1.0
    )


def test_unknown_metric_raises() -> None:
    with pytest.raises(ValueError):
        HeuristicJudge().score("bogus")


def test_parse_score() -> None:
    assert _parse_score("Score: 0.8") == 0.8
    assert _parse_score("no number here") == 0.0
    assert _parse_score("1.5") == 1.0  # clamped to [0,1]
