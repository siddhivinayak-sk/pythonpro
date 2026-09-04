"""Configuration loading: layered YAML + environment-variable expansion + typed validation.

Precedence (highest wins): process env (via ``${VAR}`` expansion in the YAML) > YAML file > model
defaults. Service-specific ``BaseSettings`` subclasses can additionally read env vars directly.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class BaseServiceSettings(BaseSettings):
    """Common settings every service shares. Reads env vars (nested via ``__``)."""

    model_config = SettingsConfigDict(env_nested_delimiter="__", extra="ignore")

    service_name: str = "ai-agent-service"
    log_level: str = "INFO"
    log_json: bool = True


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dict. Missing files yield an empty dict."""
    import yaml

    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(
            f"Config file {p} must contain a mapping at the top level, got {type(data).__name__}"
        )
    return data


def _expand_env(obj: Any) -> Any:
    """Recursively expand ``${VAR}`` / ``$VAR`` references in string values using the environment."""
    if isinstance(obj, str):
        return os.path.expandvars(obj)
    if isinstance(obj, list):
        return [_expand_env(item) for item in obj]
    if isinstance(obj, dict):
        return {key: _expand_env(value) for key, value in obj.items()}
    return obj


def load_config[ModelT: BaseModel](
    model_cls: type[ModelT], yaml_path: str | Path | None = None
) -> ModelT:
    """Load ``yaml_path`` (if given), expand env references, and validate into ``model_cls``."""
    raw = load_yaml(yaml_path) if yaml_path else {}
    raw = _expand_env(raw)
    return model_cls.model_validate(raw)
