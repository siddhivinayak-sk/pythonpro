# RAG Application (`services/rag`)

Multi-format ingestion + configurable vector store / embeddings / chunking + retrieval API + cron indexer.
Full design: [`docs/subprojects/rag.md`](../../docs/subprojects/rag.md).

> **Status: Phase 2 complete.** The full pipeline is implemented and tested end-to-end (index a directory
> → retrieve with citations). The dependency-free path (in-memory store + hashing embedder + native
> loaders) runs with no external services and is what the automated tests cover. Production adapters are
> lazy-loaded: vector stores (`pgvector`, `chroma`, `qdrant`, `milvus`), embedding providers
> (`huggingface`, `ollama`, `openai`, `azure_openai`, `bedrock`, `postgresml`), and Docling. Live backends
> (a real Postgres/Qdrant/Milvus or an embeddings API) are exercised via the Docker profiles and the
> integration tests (`RUN_INTEGRATION=1`).

## What's implemented

| Area | Details |
|------|---------|
| **Vector stores** | `memory` (tested), `pgvector` (default, production), `chroma` (embedded), `qdrant`, `milvus`. Selected by config; backend SDKs are lazy-imported and needed only for the chosen store. |
| **Embeddings** | Configurable provider + dimension: `hashing` (dep-free), `huggingface` (sentence-transformers, Matryoshka truncation), `ollama`, `openai`, `azure_openai`, `bedrock` (via the shared core factory), and `postgresml` (embeddings computed in-DB via `pgml.embed()`). Dimension validated against the collection. |
| **Chunking** | `fixed`, `recursive`, `sentence`, `markdown` (structure-aware). |
| **Loaders** | Native `text`/`markdown`/`html`; Docling (PDF/Office/images w/ OCR) via the `ingestion` extra. Registry maps extensions → loaders. |
| **Scanning** | Recursive directory scan with include/exclude globs + content-hash change detection. |
| **Indexer** | `incremental` (skips unchanged, removes deleted) and `full` modes; idempotent; per-run stats. |
| **Scheduler** | APScheduler cron jobs per collection. |
| **Retrieval** | Dense vector search; **hybrid** (dense + sparse fused with Reciprocal Rank Fusion; sparse = in-process BM25 or persistent PostgreSQL FTS); optional **cross-encoder reranking**; metadata filters (equality + `$contains`); per-collection `distance`. |
| **API** | `/v1/collections`, `/v1/retrieve` (cited hits), `/v1/index/run`, `/v1/index/status`. |

## Run locally (no external services)

```bash
uv sync --all-packages
uv run --package ai-agent-rag uvicorn ai_agent_rag.app:app --reload --port 8081
# defaults to an in-memory store + hashing embedder + a "default" collection
```

Index a directory and query it:

```bash
# point the default collection at a folder, or supply a full config (see below), then:
curl -X POST "http://localhost:8081/v1/index/run?collection=default"
curl -X POST http://localhost:8081/v1/retrieve -H "content-type: application/json" \
  -d '{"collection":"default","query":"parental leave","k":5}'
```

## Production config

Supply a pipeline YAML via `RAG_CONFIG_FILE` (see [`config/rag.example.yaml`](config/rag.example.yaml)
and docs/subprojects/rag.md §10). It defines the vector store (pgvector by default), embedding profiles
(model + dimension), chunker profiles, data sources, collections, and cron indexer jobs.

```bash
$env:RAG_CONFIG_FILE = "services/rag/config/rag.example.yaml"   # PowerShell
$env:RAG_PGVECTOR_DSN = "postgresql+psycopg://ai:pass@localhost:5432/rag"
```

Operational toggles (env):

- `RAG_ENABLE_SCHEDULER` (default `true`) — run the in-process APScheduler cron indexer for configured
  `indexer.jobs`. Set `false` on API-only replicas so jobs run only on the indexer worker.
- `RAG_ENABLE_DOCLING` (default `false`) — enable PDF/Office/image ingestion (needs the `ingestion`
  extra). When off, unsupported file types are skipped at scan time.

> **Retrieval modes:** dense by default; send `mode: hybrid` to fuse dense + sparse results with
> Reciprocal Rank Fusion. The sparse backend is `retrieval.sparse_backend`: `bm25` (default, in-process,
> dependency-free) or `pgfts` (persistent PostgreSQL full-text search reusing the pgvector DSN — durable
> and multi-replica). Enable cross-encoder reranking with `retrieval.rerank_model` or per-request
> `rerank: true`. Metadata filters support equality + `$contains`; the `distance` metric is honored on
> all backends.

Install the backend/ingestion extras as needed:

```bash
uv sync --package ai-agent-rag --extra pgvector --extra ingestion --extra embeddings-hf
# hybrid needs no extra (BM25 is dependency-free); cross-encoder reranking needs `rerank`.
```

Or run the whole stack: `docker compose --profile rag up` (from `deploy/`) — brings up the API, indexer,
and pgvector.

## Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/healthz`, `/readyz` | liveness / readiness |
| GET | `/v1/collections` | list collections + embedding/chunker/dimension/count |
| POST | `/v1/retrieve` | embed query → vector search → cited hits |
| POST | `/v1/index/run?collection=&mode=` | run an index job (incremental/full) |
| GET | `/v1/index/status?collection=` | last run + vector count |

## Tests

```bash
uv run pytest services/rag
```

Covers chunkers, the hashing embedder, the in-memory store (incl. filters), loaders, the scanner +
change detection, the sqlite index store, the scheduler, the full indexer pipeline (incremental / full /
deletions), and the API end-to-end (index a temp directory → retrieve with citations).
