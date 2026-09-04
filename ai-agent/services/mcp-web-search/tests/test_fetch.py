"""Tests for SSRF-safe fetching + readability extraction (offline; DNS and httpx are faked)."""

from __future__ import annotations

import ipaddress

import httpx
import pytest
from ai_agent_mcp_web_search import fetch as fetchmod
from ai_agent_mcp_web_search.fetch import (
    FetchError,
    SsrfError,
    extract_text,
    safe_fetch,
    validate_url,
)


# --- validate_url (SSRF guard) ------------------------------------------------
def test_validate_rejects_non_http_scheme() -> None:
    with pytest.raises(SsrfError):
        validate_url("ftp://example.com/x")


def test_validate_rejects_missing_host() -> None:
    with pytest.raises(SsrfError):
        validate_url("http:///nohost")


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1/",
        "http://localhost/",
        "http://10.0.0.1/",
        "http://192.168.1.10/",
        "http://169.254.169.254/latest/meta-data/",  # cloud metadata endpoint
    ],
)
def test_validate_blocks_internal_addresses(url: str) -> None:
    with pytest.raises(SsrfError):
        validate_url(url)


def test_validate_allows_public_when_resolved_public(monkeypatch) -> None:
    monkeypatch.setattr(fetchmod.socket, "getaddrinfo", _fake_getaddrinfo)
    assert validate_url("https://example.com/page") == "https://example.com/page"


def test_validate_allow_private_bypasses_checks() -> None:
    assert validate_url("http://127.0.0.1/", allow_private=True) == "http://127.0.0.1/"


# --- extraction ---------------------------------------------------------------
def test_extract_text_strips_boilerplate_and_keeps_article() -> None:
    html = (
        "<html><head><style>.x{}</style></head><body>"
        "<nav>menu home about</nav>"
        "<script>evil()</script>"
        "<article><h1>Title</h1><p>Hello readable world.</p></article>"
        "<footer>copyright</footer></body></html>"
    )
    text = extract_text(html)
    assert "Hello readable world." in text
    assert "evil()" not in text
    assert "menu home about" not in text
    assert "copyright" not in text


# --- safe_fetch ---------------------------------------------------------------
def _fake_getaddrinfo(host, port, *args, **kwargs):
    try:
        ipaddress.ip_address(host)
        ip = host  # IP literal: keep it so internal ranges are still caught
    except ValueError:
        ip = "93.184.216.34"  # public
    return [(2, 1, 6, "", (ip, port))]


class _FakeResp:
    def __init__(self, status_code=200, headers=None, content=b"", encoding="utf-8"):
        self.status_code = status_code
        self.headers = headers or {}
        self.content = content
        self.encoding = encoding

    @property
    def is_redirect(self) -> bool:
        return self.status_code in (301, 302, 303, 307, 308)


def _fake_client(responder):
    class _C:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def get(self, url):
            return responder(url)

    return _C


def test_safe_fetch_extracts_html(monkeypatch) -> None:
    monkeypatch.setattr(fetchmod.socket, "getaddrinfo", _fake_getaddrinfo)
    html = b"<html><body><nav>nav</nav><article>Main content here.</article></body></html>"
    responder = lambda url: _FakeResp(200, {"content-type": "text/html; charset=utf-8"}, html)  # noqa: E731
    monkeypatch.setattr(httpx, "Client", _fake_client(responder))

    result = safe_fetch("https://example.com/a", max_chars=1000)
    assert result.status_code == 200
    assert result.content_type == "text/html"
    assert "Main content here." in result.text
    assert "nav" not in result.text


def test_safe_fetch_rejects_unsupported_content_type(monkeypatch) -> None:
    monkeypatch.setattr(fetchmod.socket, "getaddrinfo", _fake_getaddrinfo)
    responder = lambda url: _FakeResp(200, {"content-type": "application/pdf"}, b"%PDF-1.7")  # noqa: E731
    monkeypatch.setattr(httpx, "Client", _fake_client(responder))
    with pytest.raises(FetchError):
        safe_fetch("https://example.com/doc.pdf")


def test_safe_fetch_revalidates_redirect_target(monkeypatch) -> None:
    monkeypatch.setattr(fetchmod.socket, "getaddrinfo", _fake_getaddrinfo)
    responder = lambda url: _FakeResp(302, {"location": "http://169.254.169.254/"}, b"")  # noqa: E731
    monkeypatch.setattr(httpx, "Client", _fake_client(responder))
    with pytest.raises(SsrfError):
        safe_fetch("https://example.com/redirect")


def test_safe_fetch_too_many_redirects(monkeypatch) -> None:
    monkeypatch.setattr(fetchmod.socket, "getaddrinfo", _fake_getaddrinfo)
    responder = lambda url: _FakeResp(302, {"location": "https://example.com/loop"}, b"")  # noqa: E731
    monkeypatch.setattr(httpx, "Client", _fake_client(responder))
    with pytest.raises(FetchError):
        safe_fetch("https://example.com/start", max_redirects=2)


def test_safe_fetch_truncates_text(monkeypatch) -> None:
    monkeypatch.setattr(fetchmod.socket, "getaddrinfo", _fake_getaddrinfo)
    body = b"<html><body><p>" + b"x" * 5000 + b"</p></body></html>"
    responder = lambda url: _FakeResp(200, {"content-type": "text/html"}, body)  # noqa: E731
    monkeypatch.setattr(httpx, "Client", _fake_client(responder))
    result = safe_fetch("https://example.com/big", max_chars=100)
    assert len(result.text) == 100
    assert result.truncated is True
