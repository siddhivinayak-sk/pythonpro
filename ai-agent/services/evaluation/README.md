# Evaluation Suite (`services/evaluation`)

Reproducible evaluation for the RAG pipeline and the conversation agent. Full design:
[`docs/subprojects/evaluation.md`](../../docs/subprojects/evaluation.md).

> **Status: Phase 5 implemented.** Retrieval metrics, judge-based generation/context metrics, agent
> metrics, gates, JSON+HTML reports, and a CI-friendly CLI are built and tested (25 tests). The default
> **heuristic judge** is dependency-free (deterministic, offline) so `eval run` works out of the box; a
> real **LLM judge** (or RAGAS/DeepEval via the `metrics` extra) plugs in behind the same interface.

## Metrics

| Suite | Metrics |
|-------|---------|
| **RAG** | Retrieval: `precision_at_k`, `recall_at_k`, `mrr`, `ndcg_at_k` (pure). Generation/context (judge): `faithfulness`, `answer_relevancy`, `context_precision`, `context_recall`, `answer_correctness`. |
| **Agent** | `tool_correctness` (vs expected tools), `task_success` (judge vs rubric), `safety` (must-not violations). |

## How it works

```
golden dataset → predictor → prediction → scoring (metrics + judge) → aggregate → gates → report
```

- **Predictors:** `ReplayPredictor` (default) scores predictions bundled in the dataset (`pred_*` fields),
  so the demo runs with no services. `LiveRagPredictor`/`LiveAgentPredictor` call the running RAG/chat
  services for real evaluation.
- **Judge:** `HeuristicJudge` (default) or `LlmJudge` (set `EVAL_JUDGE_KIND=llm` + a connection). RAGAS /
  DeepEval can be plugged behind the `Judge` interface.
- **Gates:** per-metric thresholds; a breach makes the CLI exit non-zero (CI quality gate). Defaults are
  modest for the heuristic judge — raise them for an LLM/RAGAS judge.

## Run

```bash
uv sync --all-packages
uv run --package ai-agent-evaluation python -m ai_agent_eval run --suite rag
uv run --package ai-agent-evaluation python -m ai_agent_eval run --suite agent --sample 1
# writes reports/<suite>_report.{json,html}; exits non-zero if a gate fails
```

Golden datasets are JSONL under [`datasets/`](datasets/). Every production failure should become a new
golden item (regression coverage).

## Configuration (env, prefix `EVAL_`)

| Var | Default | Notes |
|-----|---------|-------|
| `EVAL_DATASETS_DIR` | `datasets` | golden dataset location |
| `EVAL_OUT_DIR` | `reports` | report output |
| `EVAL_K` | `5` | retrieval cutoff |
| `EVAL_JUDGE_KIND` | `heuristic` | `heuristic` or `llm` |
| `EVAL_JUDGE_CONNECTION` / `EVAL_JUDGE_MODEL` | — | LLM judge model (with `EVAL_LLM_CONFIG_FILE`) |
| `EVAL_RAG_API_BASE_URL` / `EVAL_CHAT_API_BASE_URL` | service URLs | for live predictors |

Gate overrides: pass `--config eval.yaml` with a `gates: {rag: {...}, agent: {...}}` section.

## Tests

```bash
uv run pytest services/evaluation
```

Covers retrieval-metric math, the heuristic judge, per-item scoring, gate evaluation, report writing, and
the harness/CLI end-to-end (scores the shipped datasets; gate failure on empty predictions).
