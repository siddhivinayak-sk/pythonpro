"""Tests for auth: local admin, sessions, and the OIDC verifier abstraction (with a fake verifier)."""

from __future__ import annotations

import pytest
from ai_agent_chat.auth import AuthError, AuthService
from ai_agent_chat.config import ChatSettings
from ai_agent_chat.db import Database


def _service(**overrides) -> AuthService:
    settings = ChatSettings(
        db_path=":memory:", admin_username="admin", admin_password="secret", **overrides
    )
    return AuthService(Database(settings), settings)


def test_login_success_and_resolve() -> None:
    svc = _service()
    token = svc.login_local("admin", "secret")
    user = svc.resolve(token)
    assert user is not None
    assert user.role == "admin"
    assert user.source == "local"


def test_login_bad_password() -> None:
    svc = _service()
    with pytest.raises(AuthError):
        svc.login_local("admin", "wrong")


def test_login_unknown_user() -> None:
    svc = _service()
    with pytest.raises(AuthError):
        svc.login_local("bob", "secret")


def test_resolve_bad_or_missing_token() -> None:
    svc = _service()
    assert svc.resolve("nonexistent") is None
    assert svc.resolve(None) is None


def test_logout_invalidates_session() -> None:
    svc = _service()
    token = svc.login_local("admin", "secret")
    svc.logout(token)
    assert svc.resolve(token) is None


def test_oidc_login_with_fake_verifier() -> None:
    class _FakeVerifier:
        def verify(self, token: str) -> dict:
            return {
                "sub": "u-123",
                "email": "alice@example.com",
                "name": "Alice",
                "realm_access": {"roles": ["admin"]},
            }

    settings = ChatSettings(db_path=":memory:", auth_mode="both")
    svc = AuthService(Database(settings), settings, oidc_verifier=_FakeVerifier())
    token = svc.login_oidc("any-jwt")
    user = svc.resolve(token)
    assert user is not None
    assert user.source == "oidc"
    assert user.role == "admin"
    assert user.subject == "oidc:u-123"
