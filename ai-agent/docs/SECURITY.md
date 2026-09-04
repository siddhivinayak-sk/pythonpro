# Security

Security posture and hardening checklist for the AI-Agent Platform. Cross-cutting design lives in
[comprehensive-analysis.md §9.1](./comprehensive-analysis.md#91-security) and
[nfr-governance.md §5–6](./nfr-governance.md#5-llm-safety--guardrails).

## Threat model (summary)

- **Untrusted content** (documents, web results, tool output) may contain **prompt injection**. It is
  inserted as clearly delimited *data*, never executed; tools are allowlisted per conversation; the system
  prompt instructs the model to treat retrieved/searched text as reference only.
- **SSRF** via `fetch_url` (MCP server): http/https only, private/loopback/link-local/reserved IPs
  refused, per-hop redirect re-validation, content-type + size caps.
- **AuthN/Z**: local admin (pbkdf2 + server-side sessions) or OIDC (Keycloak, JWKS validation). Functional
  endpoints require a session; only health + `/metrics` are unauthenticated (keep the latter internal).
- **Secrets**: injected via env / secret manager; never logged; the UI never receives provider keys.
- **Abuse**: per-client rate limiting (token bucket) on the services; upload type/size caps.

## Hardening checklist (before shared/prod use)

- [ ] Set real secrets (`deploy/.env`): DB passwords, `OIDC_CLIENT_SECRET`, `ADMIN_PASSWORD(_HASH)`,
      Langfuse/Grafana passwords. Rotate regularly; prefer a secret manager over `.env`.
- [ ] Put services behind a TLS-terminating reverse proxy; publish only intended ports.
- [ ] Enable OIDC (`CHAT_AUTH_MODE=oidc` or `both`) for real users; keep local admin as break-glass.
- [ ] Enable rate limits (`CHAT_RATE_LIMIT_PER_MINUTE`, `RAG_RATE_LIMIT_PER_MINUTE`).
- [ ] Keep `/metrics` and internal services on a private network; do not expose publicly.
- [ ] Turn on output guards (moderation/PII redaction) per policy (nfr-governance §5) — pluggable.
- [ ] Set conversation/trace retention + per-user delete per your compliance needs (nfr-governance §6).
- [ ] Run dependency + image scans in CI (pip-audit / Trivy) and pin/lock dependencies.
- [ ] Review the RAG/MCP data flows for the corpora you index (data classification, residency).

## Known limitations

- SSRF pre-resolution checks don't fully defeat DNS-rebinding — pin the resolved IP or use an egress proxy
  in hostile environments.
- MCP bearer-token auth is config-reserved; enforce auth at the edge (reverse proxy) until wired.
- Rate limiting is per-process (best-effort); use a shared limiter (e.g., Redis) for multi-replica fairness.

## Reporting

Report vulnerabilities privately to the maintainers; do not open public issues for security problems.
