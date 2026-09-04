"""Tests for password hashing + token generation."""

from __future__ import annotations

from ai_agent_chat.security import hash_password, new_session_token, verify_password


def test_hash_verify_roundtrip() -> None:
    encoded = hash_password("s3cret")
    assert verify_password("s3cret", encoded)
    assert not verify_password("wrong", encoded)


def test_hash_uses_random_salt() -> None:
    assert hash_password("pw") != hash_password("pw")


def test_verify_rejects_malformed_encoding() -> None:
    assert not verify_password("pw", "not-a-valid-hash")


def test_tokens_are_unique() -> None:
    assert new_session_token() != new_session_token()
