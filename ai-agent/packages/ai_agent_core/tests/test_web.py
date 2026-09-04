"""Tests for the shared web hardening + observability middleware."""

from __future__ import annotations

from ai_agent_core.web import install_observability
from starlette.applications import Starlette
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient


def _client(rate_limit: int = 0) -> TestClient:
    async def hello(request):
        return PlainTextResponse("hi")

    async def health(request):
        return PlainTextResponse("ok")

    app = Starlette(routes=[Route("/hello", hello), Route("/healthz", health)])
    install_observability(app, service_name="test", rate_limit_per_minute=rate_limit)
    return TestClient(app)


def test_security_headers_present() -> None:
    resp = _client().get("/hello")
    assert resp.headers["x-content-type-options"] == "nosniff"
    assert resp.headers["x-frame-options"] == "DENY"
    assert resp.headers["referrer-policy"] == "no-referrer"


def test_metrics_endpoint_exposes_counters() -> None:
    client = _client()
    client.get("/hello")
    resp = client.get("/metrics")
    assert resp.status_code == 200
    assert "http_requests_total" in resp.text


def test_rate_limit_returns_429_after_limit() -> None:
    client = _client(rate_limit=2)
    assert client.get("/hello").status_code == 200
    assert client.get("/hello").status_code == 200
    assert client.get("/hello").status_code == 429


def test_health_is_exempt_from_rate_limit() -> None:
    client = _client(rate_limit=1)
    for _ in range(5):
        assert client.get("/healthz").status_code == 200
