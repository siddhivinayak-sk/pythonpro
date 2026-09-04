"""Tests for the search-provider abstraction (no network, no MCP SDK needed)."""

from __future__ import annotations

import pytest
from ai_agent_mcp_web_search.config import McpSettings
from ai_agent_mcp_web_search.providers import (
    SearchResult,
    SearxngProvider,
    StaticProvider,
    build_providers,
    reciprocal_rank_fusion,
    search_merged,
    search_with_fallback,
)


def _result(n: int) -> SearchResult:
    return SearchResult(title=f"t{n}", url=f"https://e/{n}", snippet=f"s{n}", source="static")


def test_static_provider_respects_max_results() -> None:
    provider = StaticProvider(results=[_result(i) for i in range(10)])
    assert len(provider.search("q", max_results=3)) == 3


def test_search_result_to_dict() -> None:
    d = _result(1).to_dict()
    assert d == {
        "title": "t1",
        "url": "https://e/1",
        "snippet": "s1",
        "source": "static",
        "published": None,
    }


def test_fallback_returns_first_non_empty() -> None:
    empty = StaticProvider(name="empty", results=[])
    full = StaticProvider(name="full", results=[_result(1)])
    out = search_with_fallback([empty, full], "q", max_results=5)
    assert len(out) == 1
    assert out[0].url == "https://e/1"


class _Failing:
    name = "failing"

    def search(self, query: str, *, max_results: int = 5, **opts: object) -> list[SearchResult]:
        raise RuntimeError("backend down")


def test_fallback_skips_failing_provider() -> None:
    out = search_with_fallback([_Failing(), StaticProvider(results=[_result(2)])], "q")
    assert out[0].url == "https://e/2"


def test_fallback_all_failing_raises() -> None:
    with pytest.raises(RuntimeError):
        search_with_fallback([_Failing(), _Failing()], "q")


def test_fallback_all_empty_returns_empty_list() -> None:
    out = search_with_fallback([StaticProvider(results=[]), StaticProvider(results=[])], "q")
    assert out == []


def test_rrf_merges_and_dedupes_by_url() -> None:
    list_a = [_result(1), _result(2), _result(3)]
    list_b = [_result(2), _result(4)]  # #2 appears in both -> should rank highest
    merged = reciprocal_rank_fusion([list_a, list_b])
    urls = [r.url for r in merged]
    assert urls[0] == "https://e/2"  # shared result wins
    assert len(urls) == len(set(urls))  # de-duplicated


def test_rrf_respects_max_results() -> None:
    merged = reciprocal_rank_fusion([[_result(i) for i in range(10)]], max_results=3)
    assert len(merged) == 3


def test_search_merged_skips_failing_provider() -> None:
    out = search_merged([_Failing(), StaticProvider(results=[_result(5)])], "q", max_results=5)
    assert [r.url for r in out] == ["https://e/5"]


def test_build_providers_defaults_to_searxng() -> None:
    providers = build_providers(McpSettings(provider_order=["searxng"]))
    assert len(providers) == 1
    assert isinstance(providers[0], SearxngProvider)


def test_build_providers_skips_disabled_duckduckgo_and_falls_back() -> None:
    providers = build_providers(
        McpSettings(provider_order=["duckduckgo"], duckduckgo_enabled=False)
    )
    assert [p.name for p in providers] == ["searxng"]


def test_build_providers_empty_order_falls_back_to_searxng() -> None:
    providers = build_providers(McpSettings(provider_order=[]))
    assert [p.name for p in providers] == ["searxng"]
