"""MCP web-search server configuration (env prefix ``MCP_``).

Complex fields (e.g. lists) can be provided as JSON in env vars, per pydantic-settings.
"""

from __future__ import annotations

from ai_agent_core.config import BaseServiceSettings
from pydantic_settings import SettingsConfigDict


class McpSettings(BaseServiceSettings):
    model_config = SettingsConfigDict(env_prefix="MCP_", env_nested_delimiter="__", extra="ignore")

    service_name: str = "mcp-web-search"
    host: str = "0.0.0.0"
    port: int = 8090
    path: str = "/mcp"

    # --- search providers ---------------------------------------------------
    # Order to try (fallback) or combine (merge). Default open-source backend: SearXNG (no API key).
    provider_order: list[str] = ["searxng"]
    search_mode: str = "fallback"  # fallback | merge (RRF across providers)
    searxng_base_url: str = "http://searxng:8080"
    duckduckgo_enabled: bool = False
    max_results: int = 5
    safe_search: int = 1  # 0 off, 1 moderate, 2 strict
    lang: str = "en"

    # --- result cache -------------------------------------------------------
    cache_ttl_seconds: int = 300
    cache_max_entries: int = 1000

    # --- fetch_url / SSRF ---------------------------------------------------
    fetch_max_chars: int = 8000
    fetch_max_bytes: int = 2_000_000
    request_timeout_seconds: int = 20
    max_redirects: int = 3
    allow_private_networks: bool = False  # SSRF guard; keep False except for trusted internal use
    user_agent: str = "ai-agent-mcp-web-search/1.0"

    # --- access control (when exposed beyond the internal network) ----------
    bearer_token: str | None = None
    rate_limit_per_minute: int = 120
