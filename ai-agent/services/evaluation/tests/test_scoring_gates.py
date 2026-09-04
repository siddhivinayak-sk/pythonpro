"""Tests for per-item scoring and gate evaluation."""

from __future__ import annotations

from ai_agent_eval.gates import evaluate_gates, merge_gates
from ai_agent_eval.judge import HeuristicJudge
from ai_agent_eval.predictors import Prediction
from ai_agent_eval.scoring import score_agent_item, score_rag_item


def test_score_rag_item() -> None:
    item = {
        "question": "parental leave policy",
        "ground_truth_answer": "paid parental leave",
        "relevant_doc_ids": ["hr/leave.pdf"],
    }
    pred = Prediction(
        answer="paid parental leave per handbook",
        contexts=["employees receive paid parental leave"],
        retrieved_ids=["hr/leave.pdf"],
    )
    scores = score_rag_item(item, pred, HeuristicJudge(), k=5)
    assert scores["recall_at_k"] == 1.0
    assert scores["mrr"] == 1.0
    assert scores["faithfulness"] > 0.5
    assert "answer_correctness" in scores


def test_score_agent_item() -> None:
    item = {
        "expected_tools": ["web_search"],
        "expected_outcome": "states the python version",
        "must_not": ["i don't know"],
    }
    pred = Prediction(answer="the python version is 3.13", tool_calls=["web_search"])
    scores = score_agent_item(item, pred, HeuristicJudge())
    assert scores["tool_correctness"] == 1.0
    assert scores["safety"] == 1.0


def test_agent_safety_violation() -> None:
    item = {"expected_tools": [], "expected_outcome": "", "must_not": ["I don't know"]}
    pred = Prediction(answer="I don't know the answer", tool_calls=[])
    assert score_agent_item(item, pred, HeuristicJudge())["safety"] == 0.0


def test_gates_pass_and_fail() -> None:
    assert evaluate_gates({"faithfulness": 0.9}, {"faithfulness": {"min": 0.5}}) == (True, [])
    passed, failures = evaluate_gates({"faithfulness": 0.3}, {"faithfulness": {"min": 0.5}})
    assert not passed
    assert len(failures) == 1


def test_gates_max_and_missing() -> None:
    assert evaluate_gates({"latency": 10}, {"latency": {"max": 5}})[0] is False
    assert evaluate_gates({}, {"faithfulness": {"min": 0.5}})[0] is True  # missing metric skipped


def test_merge_gates_override() -> None:
    gates = merge_gates("rag", {"faithfulness": {"min": 0.99}})
    assert gates["faithfulness"]["min"] == 0.99
