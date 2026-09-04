"""Evaluation configuration (env prefix ``EVAL_``)."""

from __future__ import annotations

from ai_agent_core.config import BaseServiceSettings
from pydantic_settings import SettingsConfigDict


class EvalSettings(BaseServiceSettings):
    model_config = SettingsConfigDict(env_prefix="EVAL_", env_nested_delimiter="__", extra="ignore")

    service_name: str = "evaluation-runner"

    # Systems under test.
    rag_api_base_url: str = "http://rag-api:8081"
    chat_api_base_url: str = "http://chat-api:8080"

    # Where golden datasets live and where reports are written.
    datasets_dir: str = "datasets"
    out_dir: str = "reports"

    # Scoring.
    k: int = 5
    judge_kind: str = "heuristic"  # heuristic | llm
    judge_connection: str | None = None
    judge_model: str | None = None
    llm_config_file: str | None = None  # to build the LLM judge via the connection registry
