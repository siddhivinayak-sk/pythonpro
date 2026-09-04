"""Password hashing and session-token helpers (standard library only).

pbkdf2-hmac-sha256 with a random salt is sufficient for the bootstrap admin credential; production SSO
should use the OIDC path. Session tokens are opaque, high-entropy strings stored server-side.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets

_ALGO = "pbkdf2_sha256"
_ITERATIONS = 240_000


def hash_password(password: str, *, iterations: int = _ITERATIONS) -> str:
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), iterations)
    return f"{_ALGO}${iterations}${salt}${dk.hex()}"


def verify_password(password: str, encoded: str) -> bool:
    try:
        algo, iterations_s, salt, expected = encoded.split("$", 3)
    except ValueError:
        return False
    if algo != _ALGO:
        return False
    dk = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), int(iterations_s)
    )
    return hmac.compare_digest(dk.hex(), expected)


def new_session_token() -> str:
    return secrets.token_urlsafe(32)
