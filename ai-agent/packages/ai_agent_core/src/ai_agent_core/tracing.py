"""Optional LLM tracing via Langfuse.

``build_langfuse_callbacks`` returns a list of LangChain callback handlers when tracing is enabled and the
``langfuse`` package + keys are available, and an empty list otherwise (missing keys, package not
installed, or handler construction failing) — so tracing is a transparent, opt-in add-on that never breaks
a request. ``apply_callbacks`` binds them to a real LangChain model via ``with_config`` while leaving
injected fakes (which have no ``with_config``) untouched, keeping the offline test path unaffected.

The Langfuse *server* ships in the observability Docker Compose profile; this module is the code that
actually sends traces to it once ``LANGFUSE_*`` credentials are configured.
"""

from __future__ import annotations

import os
from typing import Any

from .telemetry.logging import get_logger

log = get_logger("tracing")


def _import_langfuse_handler() -> type[Any] | None:
    """Return the Langfuse LangChain ``CallbackHandler`` class across SDK versions, or ``None``."""
    try:  # Langfuse v3+
        from langfuse.langchain import CallbackHandler

        return CallbackHandler
    except Exception:  # pragma: no cover - exercised only with the package installed
        pass
    try:  # Langfuse v2
        from langfuse.callback import CallbackHandler

        return CallbackHandler
    except Exception:  # pragma: no cover
        return None


def build_langfuse_callbacks(
    *,
    enabled: bool,
    public_key: str | None = None,
    secret_key: str | None = None,
    host: str | None = None,
) -> list[Any]:
    """Build Langfuse callback handlers, or ``[]`` when disabled/unconfigured/unavailable.

    Keys fall back to the standard ``LANGFUSE_PUBLIC_KEY`` / ``LANGFUSE_SECRET_KEY`` / ``LANGFUSE_HOST``
    environment variables so the module works with the conventional Langfuse configuration.
    """
    if not enabled:
        return []
    pk = public_key or os.environ.get("LANGFUSE_PUBLIC_KEY")
    sk = secret_key or os.environ.get("LANGFUSE_SECRET_KEY")
    resolved_host = host or os.environ.get("LANGFUSE_HOST")
    if not (pk and sk):
        log.warning("tracing_enabled_but_unconfigured", detail="missing LANGFUSE public/secret key")
        return []

    handler_cls = _import_langfuse_handler()
    if handler_cls is None:
        log.warning(
            "tracing_enabled_but_langfuse_missing", detail="pip install langfuse to enable tracing"
        )
        return []

    kwargs: dict[str, Any] = {"public_key": pk, "secret_key": sk}
    if resolved_host:
        kwargs["host"] = resolved_host
    try:
        handler = handler_cls(**kwargs)
    except Exception as exc:  # pragma: no cover - depends on installed SDK/network
        # Newer handlers read config from the environment and take no kwargs; retry bare.
        try:
            handler = handler_cls()
        except Exception:
            log.warning("tracing_handler_init_failed", error=str(exc))
            return []
    log.info("tracing_enabled", provider="langfuse", host=resolved_host or "default")
    return [handler]


def apply_callbacks(model: Any, callbacks: list[Any]) -> Any:
    """Bind callbacks to a LangChain model via ``with_config``; return the model unchanged otherwise."""
    if callbacks and hasattr(model, "with_config"):
        return model.with_config({"callbacks": list(callbacks)})
    return model
