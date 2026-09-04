"""Tests for the optional Langfuse tracing helpers (no langfuse package/network required)."""

from __future__ import annotations

import pytest
from ai_agent_core.tracing import apply_callbacks, build_langfuse_callbacks


def test_disabled_returns_empty() -> None:
    assert build_langfuse_callbacks(enabled=False, public_key="pk", secret_key="sk") == []


def test_enabled_without_keys_returns_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    assert build_langfuse_callbacks(enabled=True) == []


class _FakeModel:
    def __init__(self) -> None:
        self.config: dict | None = None

    def with_config(self, cfg: dict):
        self.config = cfg
        return self


def test_apply_callbacks_binds_when_present() -> None:
    model = _FakeModel()
    out = apply_callbacks(model, ["cb"])
    assert out is model
    assert model.config == {"callbacks": ["cb"]}


def test_apply_callbacks_noop_when_empty() -> None:
    model = _FakeModel()
    assert apply_callbacks(model, []) is model
    assert model.config is None


def test_apply_callbacks_noop_without_with_config() -> None:
    obj = object()
    assert apply_callbacks(obj, ["cb"]) is obj
