# ADR 0006 — Dependency-free core with lazy production adapters

- **Status:** Accepted (2026-09)
- **Context:** Heavy dependencies (pgvector/psycopg, sentence-transformers, Docling, chromadb, the MCP SDK)
  and external services (a live database, model downloads) make tests slow, flaky, or impossible in CI.
- **Decision:** For each abstraction, ship an always-available, dependency-free implementation used by
  tests and local dev — `HashingEmbedder`, `InMemoryVectorStore`, native loaders, sqlite index store — and
  make production adapters **optional extras that are imported lazily**. Tool/agent logic is separated from
  I/O so it can be tested with injected fakes.
- **Consequences:** The full pipeline is testable offline and CI stays fast/deterministic; the automated
  suite proves the architecture end-to-end on the in-memory path. Trade-off: production-only adapters
  (notably pgvector) are not exercised by the in-process suite and need integration testing against real
  services (Docker profiles / Phase 6 hardening).
