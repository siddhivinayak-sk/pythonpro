"""Workspace-wide pytest config: skip ``@pytest.mark.integration`` tests unless RUN_INTEGRATION=1.

Integration tests exercise the real-service paths (pgvector, MCP, live LLM providers) that the default
suite covers with fakes. Run them with services up: ``RUN_INTEGRATION=1 uv run pytest -m integration``.
"""

from __future__ import annotations

import os

import pytest


def pytest_collection_modifyitems(config, items) -> None:
    if os.environ.get("RUN_INTEGRATION"):
        return
    skip = pytest.mark.skip(reason="integration test (set RUN_INTEGRATION=1 with services up)")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip)
