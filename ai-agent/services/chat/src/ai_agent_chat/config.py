"""Chat service configuration (env prefix ``CHAT_``).

Covers auth (local admin + optional OIDC), the lightweight app database (SQLite default or DuckDB,
configurable location), uploads, and per-request defaults. The LLM connections are described by a separate
YAML (``CHAT_LLM_CONFIG_FILE``) validated into ``ai_agent_core.LLMConfig``.
"""

from __future__ import annotations

from ai_agent_core.config import BaseServiceSettings
from pydantic_settings import SettingsConfigDict


class ChatSettings(BaseServiceSettings):
    model_config = SettingsConfigDict(env_prefix="CHAT_", env_nested_delimiter="__", extra="ignore")

    service_name: str = "chat-api"
    host: str = "0.0.0.0"
    port: int = 8080
    public_url: str = "http://localhost:8080"

    # --- LLM connections ---------------------------------------------------
    llm_config_file: str | None = None

    # --- downstream services (Phase 4) -------------------------------------
    rag_api_base_url: str | None = None
    # Operator-configured MCP server URLs this deployment can use. The UI lists these; a user/conversation
    # enables a subset via the ``mcp_servers`` setting. (env: CHAT_MCP_SERVERS as a JSON array)
    mcp_servers: list[str] = []

    # --- auth --------------------------------------------------------------
    auth_mode: str = "local"  # local | oidc | both
    admin_username: str = "admin"
    admin_password: str | None = None  # dev convenience; hashed at startup
    admin_password_hash: str | None = None  # production: supply a pre-computed pbkdf2 hash
    session_ttl_hours: int = 720

    oidc_issuer: str | None = None
    oidc_client_id: str | None = None
    oidc_client_secret: str | None = None
    oidc_redirect_uri: str | None = None
    oidc_audience: str | None = None

    # --- app database (lightweight, configurable) --------------------------
    db_engine: str = "sqlite"  # sqlite | duckdb
    db_path: str = "chat.db"  # ":memory:" for ephemeral/dev; a file path otherwise

    # --- uploads / vision --------------------------------------------------
    uploads_dir: str = "./data/chat/uploads"
    max_upload_mb: int = 20

    # --- per-request defaults ---------------------------------------------
    default_temperature: float = 0.7
    default_memory_window: int = 10  # "past messages included"
    default_theme: str = "system"

    # --- CORS (dev) --------------------------------------------------------
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]

    # --- hardening ---------------------------------------------------------
    # Per-client requests/minute (0 disables). Health + /metrics are exempt.
    rate_limit_per_minute: int = 0

    # --- audio (STT/TTS) ---------------------------------------------------
    audio_enabled: bool = False
    stt_provider: str | None = None  # e.g. "openai" | "whisper-local" (pluggable)
    tts_provider: str | None = None
    tts_default_voice: str = "default"

    # --- context files -----------------------------------------------------
    context_char_cap: int = 4000  # per-file truncation when injected into the prompt

    # --- tracing (Langfuse) ------------------------------------------------
    tracing_enabled: bool = False
    langfuse_public_key: str | None = None  # falls back to LANGFUSE_PUBLIC_KEY env
    langfuse_secret_key: str | None = None  # falls back to LANGFUSE_SECRET_KEY env
    langfuse_host: str | None = None  # falls back to LANGFUSE_HOST env

    # --- output/input moderation ------------------------------------------
    moderation_enabled: bool = False
    # Case-insensitive regex/substring patterns; a match flags the text (dep-free default provider).
    moderation_blocklist: list[str] = []
    moderation_message: str = "I can't help with that request."

    # --- response cache ----------------------------------------------------
    cache_mode: str = "off"  # off | exact | semantic
    cache_similarity_threshold: float = 0.95  # semantic mode: min cosine similarity for a hit
    cache_max_entries: int = 512
