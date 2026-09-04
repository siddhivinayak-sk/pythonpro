"""Tests for the tools layer (structured envelopes, caching, error handling)."""

from __future__ import annotations

from ai_agent_mcp_web_search.config import McpSettings
from ai_agent_mcp_web_search.fetch import FetchResult, SsrfError
from ai_agent_mcp_web_search.providers import SearchResult, StaticProvider
from ai_agent_mcp_web_search.tools import WebSearchTools


def _tools(providers=None, fetcher=None) -> WebSearchTools:
    default = [StaticProvider(results=[SearchResult("T", "https://e/1", "snip", "static")])]
    return WebSearchTools(McpSettings(), providers=providers or default, fetcher=fetcher)


def test_web_search_returns_envelope() -> None:
    out = _tools().web_search("q", max_results=3)
    assert out["query"] == "q"
    assert out["count"] == 1
    assert out["results"][0]["url"] == "https://e/1"
    assert out["cached"] is False


def test_web_search_cache_hit_on_second_call() -> None:
    tools = _tools()
    assert tools.web_search("q")["cached"] is False
    assert tools.web_search("q")["cached"] is True


def test_web_search_error_is_structured() -> None:
    class _Failing:
        name = "failing"

        def search(self, query, *, max_results=5, **opts):
            raise RuntimeError("backend down")

    out = _tools(providers=[_Failing()]).web_search("q")
    assert out["results"] == []
    assert "error" in out
    assert "backend down" in out["error"]


def test_fetch_url_success_returns_text() -> None:
    def fetcher(url: str, max_chars: int) -> FetchResult:
        return FetchResult(
            url=url, status_code=200, content_type="text/html", text="hello", truncated=False
        )

    out = _tools(fetcher=fetcher).fetch_url("https://example.com")
    assert out["text"] == "hello"
    assert out["status_code"] == 200


def test_fetch_url_ssrf_error_is_structured() -> None:
    def fetcher(url: str, max_chars: int) -> FetchResult:
        raise SsrfError("host resolves to blocked address 127.0.0.1")

    out = _tools(fetcher=fetcher).fetch_url("http://127.0.0.1")
    assert out["url"] == "http://127.0.0.1"
    assert "blocked" in out["error"]


def test_max_results_capped_to_settings() -> None:
    class _Recording:
        name = "rec"
        seen: int | None = None

        def search(self, query, *, max_results=5, **opts):
            _Recording.seen = max_results
            return []

    tools = WebSearchTools(
        McpSettings(max_results=5), providers=[_Recording()], fetcher=lambda u, m: None
    )
    tools.web_search("q", max_results=50)
    assert _Recording.seen == 5


def test_news_search_returns_envelope() -> None:
    out = _tools().news_search("q")
    assert out["query"] == "q"
    assert "results" in out
