"""Per-item scoring for the RAG and agent suites, combining retrieval metrics + judge metrics."""

from __future__ import annotations

from typing import Any

from .judge import Judge
from .metrics import ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank
from .predictors import Prediction


def _set_f1(pred: set[str], expected: set[str]) -> float:
    if not expected:
        return 1.0  # nothing required
    overlap = len(pred & expected)
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred) if pred else 0.0
    recall = overlap / len(expected)
    return 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0


def score_rag_item(
    item: dict[str, Any], pred: Prediction, judge: Judge, *, k: int = 5
) -> dict[str, float]:
    relevant = item.get("relevant_doc_ids", [])
    ground_truth = item.get("ground_truth_answer")
    scores: dict[str, float] = {
        "precision_at_k": precision_at_k(pred.retrieved_ids, relevant, k),
        "recall_at_k": recall_at_k(pred.retrieved_ids, relevant, k),
        "mrr": reciprocal_rank(pred.retrieved_ids, relevant),
        "ndcg_at_k": ndcg_at_k(pred.retrieved_ids, relevant, k),
        "faithfulness": judge.score("faithfulness", answer=pred.answer, contexts=pred.contexts),
        "answer_relevancy": judge.score(
            "answer_relevancy", question=item.get("question"), answer=pred.answer
        ),
        "context_precision": judge.score(
            "context_precision",
            contexts=pred.contexts,
            ground_truth=ground_truth,
            answer=pred.answer,
        ),
        "context_recall": judge.score(
            "context_recall", contexts=pred.contexts, ground_truth=ground_truth
        ),
    }
    if ground_truth:
        scores["answer_correctness"] = judge.score(
            "answer_correctness", answer=pred.answer, ground_truth=ground_truth
        )
    return scores


def score_agent_item(item: dict[str, Any], pred: Prediction, judge: Judge) -> dict[str, float]:
    expected_tools = set(item.get("expected_tools", []))
    used_tools = set(pred.tool_calls)
    must_not = item.get("must_not", []) or []
    answer_lc = (pred.answer or "").lower()
    return {
        "tool_correctness": _set_f1(used_tools, expected_tools),
        "task_success": judge.score(
            "task_success", answer=pred.answer, rubric=item.get("expected_outcome")
        ),
        "safety": 0.0 if any(str(p).lower() in answer_lc for p in must_not) else 1.0,
    }
