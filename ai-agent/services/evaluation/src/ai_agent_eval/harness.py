"""Evaluation harness: load a golden dataset → predict → score → aggregate → apply gates.

Predictor and judge are injectable. By default it replays predictions bundled in the dataset (so
``eval run`` scores out-of-the-box) and uses the dependency-free heuristic judge; swap in live predictors
and an LLM/RAGAS judge for production evaluation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import EvalSettings
from .gates import evaluate_gates, merge_gates
from .judge import HeuristicJudge, Judge
from .metrics import mean
from .predictors import Predictor, ReplayPredictor
from .scoring import score_agent_item, score_rag_item

VALID_SUITES = ("rag", "agent")


@dataclass
class EvalResult:
    suite: str
    total: int
    passed: bool
    metrics: dict[str, float] = field(default_factory=dict)
    gate_failures: list[str] = field(default_factory=list)
    per_item: list[dict[str, Any]] = field(default_factory=list)
    judge: str = "heuristic"

    def summary(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        lines = [f"[{status}] suite={self.suite} items={self.total} judge={self.judge}"]
        for name, value in sorted(self.metrics.items()):
            lines.append(f"  {name}: {value:.3f}")
        lines.extend(f"  GATE: {f}" for f in self.gate_failures)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite": self.suite,
            "total": self.total,
            "passed": self.passed,
            "judge": self.judge,
            "metrics": self.metrics,
            "gate_failures": self.gate_failures,
            "per_item": self.per_item,
        }


def load_dataset(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    items: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            items.append(json.loads(line))
    return items


def dataset_path(suite: str, settings: EvalSettings) -> Path:
    return Path(settings.datasets_dir) / f"{suite}_golden.jsonl"


def _gates_from_config(config_path: str | None, suite: str) -> dict[str, dict[str, float]]:
    override = None
    if config_path and Path(config_path).exists():
        import yaml

        data = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
        override = (data.get("gates") or {}).get(suite)
    return merge_gates(suite, override)


def _aggregate(per_item: list[dict[str, Any]]) -> dict[str, float]:
    keys: set[str] = set()
    for row in per_item:
        keys.update(row["scores"].keys())
    return {
        key: mean(row["scores"][key] for row in per_item if key in row["scores"]) for key in keys
    }


def run_suite(
    suite: str,
    *,
    config_path: str | None = None,
    sample: int | None = None,
    settings: EvalSettings | None = None,
    predictor: Predictor | None = None,
    judge: Judge | None = None,
    gates: dict[str, dict[str, float]] | None = None,
) -> EvalResult:
    if suite not in VALID_SUITES:
        raise ValueError(f"unknown suite '{suite}', expected one of {VALID_SUITES}")

    settings = settings or EvalSettings()
    predictor = predictor or ReplayPredictor()
    judge = judge or HeuristicJudge()
    gates = gates if gates is not None else _gates_from_config(config_path, suite)

    items = load_dataset(dataset_path(suite, settings))
    if sample is not None:
        items = items[:sample]

    per_item: list[dict[str, Any]] = []
    for item in items:
        pred = predictor.predict(item)
        if suite == "rag":
            scores = score_rag_item(item, pred, judge, k=settings.k)
        else:
            scores = score_agent_item(item, pred, judge)
        per_item.append({"id": item.get("id"), "scores": scores})

    aggregate = _aggregate(per_item)
    passed, failures = (evaluate_gates(aggregate, gates)) if per_item else (True, [])
    return EvalResult(
        suite=suite,
        total=len(items),
        passed=passed,
        metrics=aggregate,
        gate_failures=failures,
        per_item=per_item,
        judge=getattr(judge, "name", "custom"),
    )


def build_default_judge(settings: EvalSettings) -> Judge:
    """Construct the judge from settings (heuristic by default; LLM judge needs a configured connection)."""
    if settings.judge_kind == "llm":
        from ai_agent_core import LLMConfig, LLMConnectionRegistry, load_config

        from .judge import LlmJudge

        config = (
            load_config(LLMConfig, settings.llm_config_file)
            if settings.llm_config_file
            else LLMConfig()
        )
        registry = LLMConnectionRegistry.from_config(config)
        model = registry.get_chat_model(
            settings.judge_connection, settings.judge_model, temperature=0.0
        )
        return LlmJudge(model)
    return HeuristicJudge()
