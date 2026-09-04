"""Authentication: local-admin bootstrap + server-side sessions, plus an OIDC verifier abstraction.

Two modes (config-selectable, can coexist):
- **local**: a bootstrap admin credential (hashed) — good for first-run and break-glass access.
- **oidc**: validate an IdP (e.g. Keycloak) token via a ``TokenVerifier`` and map claims to a user.

Sessions are opaque tokens stored in the DB; the ``current_user`` dependency accepts a Bearer header or a
``session`` cookie.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from sqlalchemy import select

from .config import ChatSettings
from .db import Database, Session, User
from .security import hash_password, new_session_token, verify_password


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)


class TokenVerifier(Protocol):
    def verify(self, token: str) -> dict[str, Any]:
        """Return validated claims (incl. 'sub') or raise on invalid token."""
        ...


class OidcVerifier:
    """Validates a JWT against the IdP's JWKS (issuer/audience/expiry). Uses ``jose`` lazily."""

    def __init__(self, issuer: str, audience: str | None = None) -> None:
        self.issuer = issuer.rstrip("/")
        self.audience = audience
        self._jwks: Any = None

    def _load_jwks(self) -> Any:
        if self._jwks is None:
            import httpx

            meta = httpx.get(f"{self.issuer}/.well-known/openid-configuration", timeout=10).json()
            self._jwks = httpx.get(meta["jwks_uri"], timeout=10).json()
        return self._jwks

    def verify(self, token: str) -> dict[str, Any]:
        from jose import jwt  # lazy (extra: auth)

        claims = jwt.decode(
            token,
            self._load_jwks(),
            audience=self.audience,
            issuer=self.issuer,
            options={"verify_aud": self.audience is not None},
        )
        return dict(claims)


class AuthError(Exception):
    pass


class AuthService:
    def __init__(
        self,
        db: Database,
        settings: ChatSettings,
        *,
        oidc_verifier: TokenVerifier | None = None,
    ) -> None:
        self.db = db
        self.settings = settings
        self._admin_hash = settings.admin_password_hash or (
            hash_password(settings.admin_password) if settings.admin_password else None
        )
        self._oidc_verifier = oidc_verifier

    # -- users / sessions --
    def _ensure_user(
        self, session, *, subject: str, source: str, role: str = "user", **fields
    ) -> User:
        user = session.scalar(select(User).where(User.subject == subject))
        if user is None:
            user = User(subject=subject, source=source, role=role, **fields)
            session.add(user)
            session.commit()
        return user

    def _create_session(self, session, user_id: str) -> str:
        token = new_session_token()
        session.add(
            Session(
                token=token,
                user_id=user_id,
                expires_at=_now() + timedelta(hours=self.settings.session_ttl_hours),
            )
        )
        session.commit()
        return token

    # -- login flows --
    def login_local(self, username: str, password: str) -> str:
        if self.settings.auth_mode not in ("local", "both"):
            raise AuthError("local login is disabled")
        if username != self.settings.admin_username or not self._admin_hash:
            raise AuthError("invalid credentials")
        if not verify_password(password, self._admin_hash):
            raise AuthError("invalid credentials")
        with self.db.session() as s:
            user = self._ensure_user(
                s, subject=username, source="local", role="admin", display_name=username
            )
            return self._create_session(s, user.id)

    def login_oidc(self, token: str) -> str:
        if self.settings.auth_mode not in ("oidc", "both") or self._oidc_verifier is None:
            raise AuthError("oidc login is not configured")
        claims = self._oidc_verifier.verify(token)
        subject = claims.get("sub")
        if not subject:
            raise AuthError("token has no subject")
        role = (
            "admin"
            if "admin" in (claims.get("realm_access", {}) or {}).get("roles", [])
            else "user"
        )
        with self.db.session() as s:
            user = self._ensure_user(
                s,
                subject=f"oidc:{subject}",
                source="oidc",
                role=role,
                email=claims.get("email"),
                display_name=claims.get("name") or claims.get("preferred_username"),
            )
            return self._create_session(s, user.id)

    # -- resolution / logout --
    def resolve(self, token: str | None) -> User | None:
        if not token:
            return None
        with self.db.session() as s:
            sess = s.get(Session, token)
            if sess is None:
                return None
            if sess.expires_at < _now():
                s.delete(sess)
                s.commit()
                return None
            return s.get(User, sess.user_id)

    def logout(self, token: str) -> None:
        with self.db.session() as s:
            sess = s.get(Session, token)
            if sess is not None:
                s.delete(sess)
                s.commit()
