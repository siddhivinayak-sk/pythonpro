# ADR 0001 — Monorepo managed as a uv workspace

- **Status:** Accepted (2026-09)
- **Context:** Four independent sub-projects plus a shared library need isolation *and* shared tooling,
  consistent dependencies, and easy cross-project refactoring.
- **Decision:** Use a single git repo (monorepo) with a **uv workspace**; each package/service has its own
  `pyproject.toml` and the workspace shares one lockfile and virtualenv. Shared logic lives in
  `packages/ai_agent_core`.
- **Consequences:** One `uv sync --all-packages` sets up everything; services still build/ship
  independently (per-service Dockerfiles + Compose profiles). Requires `--all-packages` for full setup
  (the root has no runtime deps). Alternative (polyrepo) was rejected for higher coordination overhead.
