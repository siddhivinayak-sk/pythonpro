"""Tests for the harness, reports, and CLI (scores the shipped demo datasets end-to-end)."""

from __future__ import annotations

from pathlib import Path

import pytest
from ai_agent_eval.cli import build_parser, main
from ai_agent_eval.config import EvalSettings
from ai_agent_eval.harness import EvalResult, run_suite
from ai_agent_eval.predictors import Prediction
from ai_agent_eval.report import write_reports

DATASETS = str(Path(__file__).resolve().parent.parent / "datasets")


def test_rag_suite_scores_and_passes() -> None:
    result = run_suite("rag", settings=EvalSettings(datasets_dir=DATASETS))
    assert result.total == 3
    assert result.passed, result.gate_failures
    assert "faithfulness" in result.metrics
    assert result.metrics["recall_at_k"] > 0.5


def test_agent_suite_scores_and_passes() -> None:
    result = run_suite("agent", settings=EvalSettings(datasets_dir=DATASETS))
    assert result.passed, result.gate_failures
    assert result.metrics["tool_correctness"] == 1.0
    assert result.metrics["safety"] == 1.0


def test_gate_fails_with_empty_predictions() -> None:
    class _Empty:
        def predict(self, item):
            return Prediction()

    result = run_suite("rag", settings=EvalSettings(datasets_dir=DATASETS), predictor=_Empty())
    assert not result.passed
    assert result.gate_failures


def test_unknown_suite_raises() -> None:
    with pytest.raises(ValueError):
        run_suite("nope")


def test_parser_accepts_run() -> None:
    args = build_parser().parse_args(["run", "--suite", "rag"])
    assert args.command == "run"
    assert args.suite == "rag"


def test_write_reports(tmp_path: Path) -> None:
    result = EvalResult(
        suite="rag",
        total=1,
        passed=True,
        metrics={"faithfulness": 0.9},
        per_item=[{"id": "r1", "scores": {"faithfulness": 0.9}}],
    )
    paths = write_reports(result, str(tmp_path))
    assert Path(paths["json"]).exists()
    html = Path(paths["html"]).read_text(encoding="utf-8")
    assert "faithfulness" in html and "PASS" in html


def test_main_run_writes_report_and_exits_zero(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("EVAL_DATASETS_DIR", DATASETS)
    monkeypatch.setenv("EVAL_OUT_DIR", str(tmp_path))
    assert main(["run", "--suite", "rag"]) == 0
    assert (tmp_path / "rag_report.json").exists()
    assert (tmp_path / "rag_report.html").exists()
