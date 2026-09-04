"""Command-line interface for the evaluation suite.

Usage:
    ai-agent-eval run --suite rag [--config eval.yaml] [--sample N]
    python -m ai_agent_eval run --suite agent
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from ai_agent_core import configure_logging, get_logger

from . import __version__
from .config import EvalSettings
from .harness import VALID_SUITES, build_default_judge, run_suite
from .report import write_reports


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ai-agent-eval", description="Evaluate the RAG pipeline or the conversation agent."
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run an evaluation suite")
    run.add_argument("--suite", choices=VALID_SUITES, required=True)
    run.add_argument("--config", default=None, help="Path to eval.yaml (Phase 5)")
    run.add_argument("--sample", type=int, default=None, help="Evaluate only the first N items")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    configure_logging(json_logs=False)
    log = get_logger("eval")
    args = build_parser().parse_args(argv)

    if args.command == "run":
        settings = EvalSettings()
        judge = build_default_judge(settings)
        result = run_suite(
            args.suite, config_path=args.config, sample=args.sample, settings=settings, judge=judge
        )
        paths = write_reports(result, settings.out_dir)
        log.info("eval_complete", suite=result.suite, items=result.total, passed=result.passed)
        print(result.summary())
        print(f"reports: {paths['json']}  {paths['html']}")
        return 0 if result.passed else 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
