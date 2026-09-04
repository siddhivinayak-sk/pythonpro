"""Shared web hardening + observability for the FastAPI services.

Provides Starlette middleware for Prometheus metrics, security headers, and a simple in-process rate
limiter, plus ``install_observability(app, ...)`` to wire them (and a ``/metrics`` endpoint) onto any
FastAPI/Starlette app. Kept out of the package ``__init__`` so non-web consumers don't import Starlette.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable

from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

# Module-level metrics (created once) to avoid duplicate-registration errors across multiple apps.
REQUESTS = Counter("http_requests_total", "HTTP requests", ["service", "method", "path", "status"])
LATENCY = Histogram(
    "http_request_duration_seconds", "HTTP request latency (s)", ["service", "method", "path"]
)

_EXEMPT_PATHS = {"/healthz", "/readyz", "/metrics"}


def _route_path(request: Request) -> str:
    route = request.scope.get("route")
    return getattr(route, "path", None) or request.url.path


class MetricsMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, service_name: str) -> None:
        super().__init__(app)
        self.service = service_name

    async def dispatch(self, request: Request, call_next: Callable):
        start = time.perf_counter()
        response = await call_next(request)
        path = _route_path(request)
        REQUESTS.labels(self.service, request.method, path, str(response.status_code)).inc()
        LATENCY.labels(self.service, request.method, path).observe(time.perf_counter() - start)
        return response


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: Callable):
        response = await call_next(request)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault("X-XSS-Protection", "0")
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Per-client token-bucket rate limit (best-effort, per process). Health/metrics are exempt."""

    def __init__(self, app, limit_per_minute: int, exempt_paths: set[str] | None = None) -> None:
        super().__init__(app)
        self.capacity = float(limit_per_minute)
        self.refill_per_sec = limit_per_minute / 60.0
        self.exempt = exempt_paths or _EXEMPT_PATHS
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def _allow(self, key: str) -> bool:
        now = time.monotonic()
        with self._lock:
            tokens, last = self._buckets.get(key, (self.capacity, now))
            tokens = min(self.capacity, tokens + (now - last) * self.refill_per_sec)
            if tokens < 1.0:
                self._buckets[key] = (tokens, now)
                return False
            self._buckets[key] = (tokens - 1.0, now)
            return True

    async def dispatch(self, request: Request, call_next: Callable):
        if request.url.path in self.exempt:
            return await call_next(request)
        key = request.client.host if request.client else "anonymous"
        if not self._allow(key):
            return JSONResponse({"detail": "rate limit exceeded"}, status_code=429)
        return await call_next(request)


async def metrics_endpoint(request: Request) -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


def install_observability(app, *, service_name: str, rate_limit_per_minute: int = 0) -> None:
    """Add metrics + security headers (+ optional rate limit) and expose ``/metrics``."""
    if rate_limit_per_minute and rate_limit_per_minute > 0:
        app.add_middleware(RateLimitMiddleware, limit_per_minute=rate_limit_per_minute)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(MetricsMiddleware, service_name=service_name)
    app.add_route("/metrics", metrics_endpoint, methods=["GET"])
