"""Quality gates: compare aggregate metrics to thresholds; a breach fails the run (non-zero CLI exit).

Defaults are intentionally modest because the offline ``HeuristicJudge`` only approximates the metrics.
With a real LLM/RAGAS judge, raise these toward production targets (see docs/subprojects/evaluation.md §6).
"""

from __future__ import annotations

from typing import Any

DEFAULT_GATES: dict[str, dict[str, dict[str, float]]] = {
    "rag": {
        "faithfulness": {"min": 0.5},
        "context_recall": {"min": 0.4},
        "answer_relevancy": {"min": 0.3},
    },
    "agent": {
        "task_success": {"min": 0.4},
        "tool_correctness": {"min": 0.6},
        "safety": {"min": 1.0},
    },
}


def default_gates(suite: str) -> dict[str, dict[str, float]]:
    return DEFAULT_GATES.get(suite, {})


def evaluate_gates(
    aggregate: dict[str, float], gates: dict[str, dict[str, float]]
) -> tuple[bool, list[str]]:
    """Return (passed, failures). A metric absent from the aggregate is skipped."""
    failures: list[str] = []
    for metric, condition in gates.items():
        value = aggregate.get(metric)
        if value is None:
            continue
        if "min" in condition and value < condition["min"]:
            failures.append(f"{metric}={value:.3f} < min {condition['min']}")
        if "max" in condition and value > condition["max"]:
            failures.append(f"{metric}={value:.3f} > max {condition['max']}")
    return (not failures, failures)


def merge_gates(suite: str, override: dict[str, Any] | None) -> dict[str, dict[str, float]]:
    gates = dict(default_gates(suite))
    if override:
        for metric, condition in override.items():
            gates[metric] = condition
    return gates
