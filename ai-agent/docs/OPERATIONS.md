# Operations Runbook

Operating the AI-Agent Platform: observability, backup/restore/DR, scaling, and common incidents. See also
[nfr-governance.md](./nfr-governance.md) (SLOs §2, backup/DR §8, capacity §9).

## Observability

Run the stack with the observability profile:

```bash
docker compose --profile all up          # includes observability
# or just the observability tooling alongside your services:
docker compose --profile observability up
```

- **Metrics:** the chat (`:8080`) and RAG (`:8081`) services expose Prometheus metrics at `/metrics`
  (`http_requests_total`, `http_request_duration_seconds`). Prometheus (`:9090`) scrapes them
  (`deploy/prometheus/prometheus.yml`); Grafana (`:3002`, admin/`$GRAFANA_ADMIN_PASSWORD`) charts them.
- **Tracing / cost:** Langfuse (`:3001`) collects LLM/RAG/agent traces (prompts, tokens, latency, cost).
  Point services at it via env when enabling tracing (a later wiring; the metrics + logs path is built).
- **Logs:** structured JSON (structlog) to stdout — ship with your log collector.

> `/metrics` is unauthenticated by design (internal scrape). Keep it on the internal network / behind the
> reverse proxy; do not expose it publicly.

## Backup & restore

| Store | Backup | Restore |
|-------|--------|---------|
| Postgres/pgvector | `pg_dump -Fc` (or WAL/PITR) | `pg_restore`; re-index if vectors lost (source docs are the source of truth) |
| Chat app DB (SQLite) | copy `CHAT_DB_PATH` file (or litestream) | copy the file back |
| RAG index-metadata DB | copy `RAG_INDEX_DB_PATH` | rebuild via a full re-index if lost |
| Uploads | back up the uploads volume | restore the volume |
| Config | version control (no secrets) + secret-manager backup | redeploy |

**Rebuild path:** the vector store is reconstructable from source documents — run a full index
(`POST /v1/index/run?collection=<name>&mode=full`). Targets: RPO ≤ 24h, RTO ≤ 1h (tune per deployment).

## Scaling

- The chat and RAG APIs are **stateless** — run multiple replicas behind a load balancer; state lives in
  Postgres/vector store/app DB.
- The RAG indexer is a scheduled/one-shot worker (APScheduler cron). Run a single indexer per collection
  to avoid overlap (it locks per collection).
- Vector search scales with the backend: pgvector → Qdrant/Milvus as volume grows (swap via config).

## Common incidents

| Symptom | Likely cause | Action |
|---------|--------------|--------|
| `/v1/retrieve` 500 | pgvector unreachable / dim mismatch | check DB; verify embedding dim == collection dim |
| Chat 502 "chat failed" | model/provider error or missing provider extra | check connection creds; install provider extra |
| 429 responses | rate limit hit | raise `*_RATE_LIMIT_PER_MINUTE` or scale out |
| MCP tools missing | MCP server down | check `--profile mcp`; server URL in chat settings |
| Eval gate fails in CI | quality regression | open the HTML report; triage lowest-scoring items |

## Quality gates in CI

`uv run --package ai-agent-evaluation python -m ai_agent_eval run --suite rag` exits non-zero on a gate
breach (see the CI workflow). Reports are written to `reports/` (JSON for trends, HTML for triage).
