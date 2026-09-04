# ADR 0004 — Custom React + FastAPI conversation front end

- **Status:** Accepted (2026-09)
- **Context:** The conversation app has highly custom requirements: admin + Keycloak auth, per-conversation
  RAG/MCP tool selection, a workflow builder, vision, and fine-grained memory/temperature/system-prompt
  controls. Options considered: fork Open WebUI, use Chainlit, or build custom.
- **Decision:** Build a **custom React + Vite + TypeScript** front end against a **FastAPI** backend. Open
  WebUI serves as a UX reference, not a base.
- **Consequences:** Full control over the bespoke UX and a reusable, headless backend API; more build
  effort than adopting an existing app. Chainlit/Open WebUI were rejected as bases because deep
  customization would fight their opinions.
