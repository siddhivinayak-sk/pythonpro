"""LLM-as-judge abstraction for generation/context metrics.

- ``HeuristicJudge`` — dependency-free, deterministic token-overlap scoring. It's the default so the suite
  runs (and tests are stable) with no model. It approximates the metrics, not replaces a real judge.
- ``LlmJudge`` — asks a chat model for a 0..1 score per metric (lazy; needs a provider).
- RAGAS / DeepEval can be plugged behind the same ``Judge`` interface via the ``metrics`` extra.

Scored metrics: faithfulness, answer_relevancy, context_precision, context_recall, answer_correctness,
task_success.
"""

from __future__ import annotations

import re
from typing import Any, Protocol, runtime_checkable

_STOPWORDS = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "was",
    "were",
    "of",
    "to",
    "in",
    "on",
    "for",
    "and",
    "or",
    "but",
    "with",
    "as",
    "by",
    "at",
    "it",
    "this",
    "that",
    "be",
    "do",
    "does",
    "how",
    "what",
    "why",
    "when",
    "i",
    "you",
    "we",
    "they",
    "he",
    "she",
    "my",
    "our",
    "your",
    "their",
    "from",
    "can",
    "will",
}
_TOKEN = re.compile(r"[a-z0-9]+")


def _tokens(text: str | None) -> set[str]:
    if not text:
        return set()
    return {t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1}


def _coverage(target: set[str], source: set[str]) -> float:
    """Fraction of target tokens present in source."""
    return len(target & source) / len(target) if target else 0.0


def _f1(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    overlap = len(a & b)
    if overlap == 0:
        return 0.0
    precision = overlap / len(a)
    recall = overlap / len(b)
    return 2 * precision * recall / (precision + recall)


@runtime_checkable
class Judge(Protocol):
    def score(
        self,
        metric: str,
        *,
        question: str | None = None,
        answer: str | None = None,
        contexts: list[str] | None = None,
        ground_truth: str | None = None,
        rubric: str | None = None,
    ) -> float: ...


class HeuristicJudge:
    name = "heuristic"

    def score(
        self,
        metric: str,
        *,
        question: str | None = None,
        answer: str | None = None,
        contexts: list[str] | None = None,
        ground_truth: str | None = None,
        rubric: str | None = None,
    ) -> float:
        ctx_tokens = _tokens(" ".join(contexts or []))
        ans_tokens = _tokens(answer)

        if metric == "faithfulness":
            return _coverage(ans_tokens, ctx_tokens)
        if metric == "answer_relevancy":
            q = _tokens(question)
            return len(ans_tokens & q) / min(len(ans_tokens), len(q)) if ans_tokens and q else 0.0
        if metric == "context_precision":
            reference = _tokens(ground_truth) or ans_tokens
            relevant = sum(1 for c in (contexts or []) if _tokens(c) & reference)
            return relevant / len(contexts) if contexts else 0.0
        if metric == "context_recall":
            return _coverage(_tokens(ground_truth), ctx_tokens)
        if metric == "answer_correctness":
            return _f1(ans_tokens, _tokens(ground_truth))
        if metric == "task_success":
            expected = _tokens(rubric)
            return len(ans_tokens & expected) / len(expected) if expected else 0.0
        raise ValueError(f"unknown judge metric: {metric}")


class LlmJudge:
    """Score with a chat model (lazy). ``model`` must expose ``.invoke(messages) -> obj.content``."""

    name = "llm"

    _PROMPTS = {
        "faithfulness": "Is the ANSWER fully supported by the CONTEXT? Reply with a number 0..1.",
        "answer_relevancy": "How well does the ANSWER address the QUESTION? Reply with a number 0..1.",
        "context_precision": "What fraction of the CONTEXT is relevant to the QUESTION? 0..1.",
        "context_recall": "Does the CONTEXT contain everything in the GROUND TRUTH? 0..1.",
        "answer_correctness": "How correct is the ANSWER vs the GROUND TRUTH? 0..1.",
        "task_success": "Did the ANSWER achieve the expected outcome (RUBRIC)? 0..1.",
    }

    def __init__(self, model: Any) -> None:
        self._model = model

    def score(self, metric: str, **fields: Any) -> float:
        from langchain_core.messages import HumanMessage

        instruction = self._PROMPTS.get(metric)
        if instruction is None:
            raise ValueError(f"unknown judge metric: {metric}")
        payload = "\n".join(f"{k.upper()}: {v}" for k, v in fields.items() if v)
        result = self._model.invoke([HumanMessage(content=f"{instruction}\n\n{payload}\n\nScore:")])
        return _parse_score(getattr(result, "content", result))


def _parse_score(text: Any) -> float:
    match = re.search(r"[01](?:\.\d+)?", str(text))
    if not match:
        return 0.0
    return max(0.0, min(1.0, float(match.group(0))))
