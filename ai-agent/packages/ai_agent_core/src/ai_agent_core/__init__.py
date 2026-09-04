"""Shared core for the AI-Agent Platform.

Exposes the configuration models, the LLM connection registry, telemetry helpers, and shared schemas.
Importing this package is cheap: LangChain provider SDKs are imported lazily by the model factory.
"""

from __future__ import annotations

from .config.base import BaseServiceSettings, load_config, load_yaml
from .config.models import (
    Capability,
    ConnectionConfig,
    LLMConfig,
    ModelDescriptor,
    Provider,
)
from .llm.registry import (
    ConnectionNotFoundError,
    LLMConnectionRegistry,
    ModelNotFoundError,
)
from .llm.types import ResolvedModel
from .telemetry.logging import configure_logging, get_logger
from .tracing import apply_callbacks, build_langfuse_callbacks

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # config
    "BaseServiceSettings",
    "load_config",
    "load_yaml",
    "Capability",
    "ConnectionConfig",
    "LLMConfig",
    "ModelDescriptor",
    "Provider",
    # llm
    "LLMConnectionRegistry",
    "ConnectionNotFoundError",
    "ModelNotFoundError",
    "ResolvedModel",
    # telemetry
    "configure_logging",
    "get_logger",
    # tracing
    "build_langfuse_callbacks",
    "apply_callbacks",
]
