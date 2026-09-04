"""Search-provider abstraction (pure logic, no MCP dependency — unit-testable offline).

SearXNG is the default open-source backend. Providers can be tried in order (``fallback``) or combined
with Reciprocal Rank Fusion (``merge``). DuckDuckGo is an optional, key-free extra.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from .config import McpSettings


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    source: str
    published: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@runtime_checkable
class SearchProvider(Protocol):
    name: str

    def search(self, query: str, *, max_results: int = 5, **opts: Any) -> list[SearchResult]: ...


class SearxngProvider:
    """Query a self-hosted SearXNG instance via its JSON API."""

    name = "searxng"

    def __init__(self, base_url: str, timeout: int = 20) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def search(self, query: str, *, max_results: int = 5, **opts: Any) -> list[SearchResult]:
        import httpx

        params: dict[str, Any] = {
            "q": query,
            "format": "json",
            "safesearch": opts.get("safe_search", 1),
        }
        if opts.get("lang"):
            params["language"] = opts["lang"]
        if opts.get("time_range"):
            params["time_range"] = opts["time_range"]  # day | week | month | year
        if opts.get("categories"):
            params["categories"] = opts["categories"]  # e.g. "news"

        resp = httpx.get(f"{self.base_url}/search", params=params, timeout=self.timeout)
        resp.raise_for_status()
        payload = resp.json()
        results: list[SearchResult] = []
        for item in payload.get("results", [])[:max_results]:
            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("content", ""),
                    source="searxng",
                    published=item.get("publishedDate"),
                )
            )
        return results


class DuckDuckGoProvider:
    """Optional, key-free provider backed by the ``ddgs`` package (lazy import)."""

    name = "duckduckgo"

    def search(self, query: str, *, max_results: int = 5, **opts: Any) -> list[SearchResult]:
        try:
            from ddgs import DDGS
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "DuckDuckGo provider requires the 'ddgs' package (extra: duckduckgo)"
            ) from exc

        results: list[SearchResult] = []
        with DDGS() as ddgs:
            for item in ddgs.text(query, max_results=max_results):
                results.append(
                    SearchResult(
                        title=item.get("title", ""),
                        url=item.get("href", ""),
                        snippet=item.get("body", ""),
                        source="duckduckgo",
                    )
                )
        return results


@dataclass
class StaticProvider:
    """Returns canned results — used for tests and offline development."""

    name: str = "static"
    results: list[SearchResult] = field(default_factory=list)

    def search(self, query: str, *, max_results: int = 5, **opts: Any) -> list[SearchResult]:
        return list(self.results[:max_results])


def search_with_fallback(
    providers: list[SearchProvider], query: str, *, max_results: int = 5, **opts: Any
) -> list[SearchResult]:
    """Try providers in order; return the first non-empty result set.

    A provider that errors is skipped. If every provider errors (none even ran successfully), the last
    error is raised. If providers ran but all returned nothing, an empty list is returned.
    """
    last_error: Exception | None = None
    any_succeeded = False
    for provider in providers:
        try:
            results = provider.search(query, max_results=max_results, **opts)
        except Exception as exc:  # noqa: BLE001 - deliberately resilient across backends
            last_error = exc
            continue
        any_succeeded = True
        if results:
            return results
    if not any_succeeded and last_error is not None:
        raise last_error
    return []


def reciprocal_rank_fusion(
    result_lists: list[list[SearchResult]], *, k: int = 60, max_results: int | None = None
) -> list[SearchResult]:
    """Combine several ranked lists into one, de-duplicating by URL (RRF)."""
    scores: dict[str, float] = {}
    best: dict[str, SearchResult] = {}
    for results in result_lists:
        for rank, item in enumerate(results):
            if not item.url:
                continue
            scores[item.url] = scores.get(item.url, 0.0) + 1.0 / (k + rank + 1)
            best.setdefault(item.url, item)
    ranked = sorted(best.values(), key=lambda r: scores[r.url], reverse=True)
    return ranked[:max_results] if max_results is not None else ranked


def search_merged(
    providers: list[SearchProvider], query: str, *, max_results: int = 5, **opts: Any
) -> list[SearchResult]:
    """Query all providers and merge their results with RRF. Failing providers are skipped."""
    lists: list[list[SearchResult]] = []
    for provider in providers:
        try:
            lists.append(provider.search(query, max_results=max_results, **opts))
        except Exception:  # noqa: BLE001 - resilient across backends
            continue
    return reciprocal_rank_fusion(lists, max_results=max_results)


def build_providers(settings: McpSettings) -> list[SearchProvider]:
    """Construct the provider chain from configuration."""
    providers: list[SearchProvider] = []
    for name in settings.provider_order:
        if name == "searxng":
            providers.append(
                SearxngProvider(settings.searxng_base_url, timeout=settings.request_timeout_seconds)
            )
        elif name == "duckduckgo" and settings.duckduckgo_enabled:
            providers.append(DuckDuckGoProvider())
    if not providers:
        providers.append(
            SearxngProvider(settings.searxng_base_url, timeout=settings.request_timeout_seconds)
        )
    return providers
