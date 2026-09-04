# AI-Agent Platform

A modular, self-hostable AI platform built with **Python**, **LangChain**, and **LangGraph**. It provides
multi-provider LLM access (OpenAI, AWS Bedrock, Ollama), a configurable RAG engine, a ChatGPT-style
conversation front end, an MCP web-search server for web grounding, and a dedicated evaluation suite.

Every capability is delivered as an **independent, isolated sub-project** so it can be run on its own or
composed together via Docker Compose.

> **Status: all six phases built and tested (GA-ready).** Shared core, MCP web-search, RAG application,
> the conversation front end (auth, streaming chat, history, settings, model switching, vision + React
> UI), integration (a tool-calling agent that grounds answers with RAG + web search and cites them, plus a
> workflow engine), the evaluation suite (scored reports + CI gates), and hardening (Prometheus metrics,
> security headers + rate limiting, observability Compose profile, CI, ops/security docs). **175 tests
> pass.** Ongoing operational work needs a live deployment (security audit, load runs, DR drills, Langfuse
> trace wiring). See the [roadmap](docs/comprehensive-analysis.md#11-phased-delivery-roadmap) and the
> [traceability matrix](docs/nfr-governance.md#1-requirement-traceability-matrix).

---

## Sub-projects

| # | Sub-project | Purpose | Runs independently |
|---|-------------|---------|--------------------|
| 1 | **RAG Application** (`services/rag`) | Ingests multi-format documents, builds embeddings into a configurable vector DB, and serves retrieval APIs. Includes a cron-driven indexer. | ✅ |
| 2 | **Conversation Front End** (`services/chat`) | Web UI + backend for login, chat, history, settings, model switching, vision, RAG/MCP selection, and workflows. | ✅ |
| 3 | **MCP Web-Search Server** (`services/mcp-web-search`) | Model Context Protocol server exposing web-grounding tools over open search engines. | ✅ |
| 4 | **Evaluation Suite** (`services/evaluation`) | Offline + CI evaluation for the RAG pipeline and the conversation agent. | ✅ |
| — | **Shared Core** (`packages/ai_agent_core`) | Reusable library: LLM connection registry, configuration, schemas, telemetry. | Library |

---

## Documentation map

Start here and drill down:

- **[Comprehensive Analysis & Design](docs/comprehensive-analysis.md)** — the master document: market
  research, vision, architecture, technology decisions, cross-cutting concerns, deployment, roadmap, risks.
- **[Non-Functional Requirements, Governance & Operability](docs/nfr-governance.md)** — SLOs, cost
  governance, LLM safety, data governance/privacy, feedback loop, caching, backup/DR, and the full
  **requirement-traceability matrix**.
- **[Architecture Decision Records](docs/adr/README.md)** — key decisions and their rationale.
- **[Operations Runbook](docs/OPERATIONS.md)** & **[Security](docs/SECURITY.md)** — observability,
  backup/DR, scaling, and the hardening checklist.
- Deep dives (per sub-project):
  - [RAG Application](docs/subprojects/rag.md)
  - [Conversation Front End](docs/subprojects/conversation-frontend.md)
  - [MCP Web-Search Server](docs/subprojects/mcp-web-search.md)
  - [Evaluation Suite](docs/subprojects/evaluation.md)
- **[Development guide](DEVELOPMENT.md)** — workspace setup, commands, containerized runs.

Each built sub-project ships its own `README.md` under `services/<name>/`.

---

## Design principles at a glance

1. **Configuration over code** — providers, models, vector stores, embedding models, chunkers, and data
   sources are all selected through configuration, not code changes.
2. **Multiple connections, many models** — you can register several OpenAI / Bedrock / Ollama connections
   at once; all their models are enumerated at runtime for the user to choose.
3. **Isolation** — each sub-project has its own dependencies, container image, and Compose profile.
4. **Observable & testable** — structured logs, tracing, and metrics everywhere; tests are a first-class
   deliverable for each sub-project.
5. **Open-source-first** — self-hostable defaults (Ollama, SearXNG, pgvector/Qdrant/Milvus, open
   embedding models) with cloud options where useful.

---

## Repository layout

```
ai-agent/
├── README.md                     # this file
├── DEVELOPMENT.md                # developer setup & commands
├── pyproject.toml                # uv workspace + tooling (ruff, mypy, pytest)
├── docs/
│   ├── comprehensive-analysis.md # master analysis & design
│   ├── nfr-governance.md         # NFRs, governance, traceability matrix
│   ├── adr/                      # architecture decision records
│   └── subprojects/              # per-sub-project deep dives
│       ├── rag.md
│       ├── conversation-frontend.md
│       ├── mcp-web-search.md
│       └── evaluation.md
├── packages/
│   └── ai_agent_core/            # shared library — LLM registry, config, telemetry, schemas  ✅
├── services/
│   ├── rag/                      # RAG application (sub-project 1)                             ✅
│   ├── chat/                     # conversation front end (sub-project 2) — backend + frontend/  ✅
│   ├── mcp-web-search/           # MCP web-search server (sub-project 3)                       ✅
│   └── evaluation/               # evaluation suite (sub-project 4)                            ✅
└── deploy/
    ├── docker-compose.yml        # single root orchestrator with per-service profiles
    ├── .env.example              # copy to .env; compose variable substitution
    └── searxng/                  # SearXNG settings for the mcp profile
```

Legend: ✅ built & tested · 🟡 partial · ⬜ designed, not yet built.

## License

Intended to follow the parent repository's MIT license unless stated otherwise per sub-project.
