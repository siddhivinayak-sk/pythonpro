"""Tool logic for the web-search MCP server.

Kept separate from the MCP wiring (``server.py``) so it can be unit-tested with injected providers and a
fake fetcher. Every method returns a consistent, model-readable dict envelope and turns failures into
structured ``error`` fields rather than raising (so the agent gets a useful message).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ai_agent_core import get_logger

from .cache import TTLCache
from .config import McpSettings
from .fetch import FetchError, FetchResult, safe_fetch
from .providers import (
    SearchProvider,
    build_providers,
    search_merged,
    search_with_fallback,
)

FetcherType = Callable[[str, int], FetchResult]


class WebSearchTools:
    def __init__(
        self,
        settings: McpSettings,
        *,
        providers: list[SearchProvider] | None = None,
        fetcher: FetcherType | None = None,
        cache: TTLCache | None = None,
    ) -> None:
        self.settings = settings
        self.providers = providers if providers is not None else build_providers(settings)
        self.fetcher = fetcher or self._default_fetcher
        self.cache = (
            cache
            if cache is not None
            else TTLCache(settings.cache_ttl_seconds, settings.cache_max_entries)
        )
        self.log = get_logger("mcp-web-search")

    def _default_fetcher(self, url: str, max_chars: int) -> FetchResult:
        return safe_fetch(
            url,
            max_chars=max_chars,
            max_bytes=self.settings.fetch_max_bytes,
            timeout=self.settings.request_timeout_seconds,
            max_redirects=self.settings.max_redirects,
            user_agent=self.settings.user_agent,
            allow_private=self.settings.allow_private_networks,
        )

    def _search(
        self,
        query: str,
        max_results: int,
        *,
        categories: str | None = None,
        time_range: str | None = None,
    ) -> dict[str, Any]:
        limit = min(max_results, self.settings.max_results)
        cache_key = (
            "search",
            query,
            limit,
            categories or "",
            time_range or "",
            self.settings.lang,
            self.settings.search_mode,
        )
        cached = self.cache.get(cache_key)
        if cached is not None:
            return {**cached, "cached": True}

        opts: dict[str, Any] = {
            "safe_search": self.settings.safe_search,
            "lang": self.settings.lang,
        }
        if categories:
            opts["categories"] = categories
        if time_range:
            opts["time_range"] = time_range

        try:
            if self.settings.search_mode == "merge":
                results = search_merged(self.providers, query, max_results=limit, **opts)
            else:
                results = search_with_fallback(self.providers, query, max_results=limit, **opts)
        except Exception as exc:  # noqa: BLE001 - surface a readable error to the agent
            self.log.warning("search_failed", query=query, error=str(exc))
            return {"query": query, "results": [], "count": 0, "error": f"search failed: {exc}"}

        payload = {"query": query, "results": [r.to_dict() for r in results], "count": len(results)}
        self.cache.set(cache_key, payload)
        return {**payload, "cached": False}

    def web_search(
        self, query: str, max_results: int = 5, time_range: str | None = None
    ) -> dict[str, Any]:
        return self._search(query, max_results, time_range=time_range)

    def news_search(
        self, query: str, max_results: int = 5, time_range: str = "week"
    ) -> dict[str, Any]:
        return self._search(query, max_results, categories="news", time_range=time_range)

    def fetch_url(self, url: str, max_chars: int = 8000) -> dict[str, Any]:
        limit = min(max_chars, self.settings.fetch_max_chars)
        try:
            result = self.fetcher(url, limit)
        except FetchError as exc:
            self.log.warning("fetch_refused", url=url, error=str(exc))
            return {"url": url, "error": str(exc)}
        except Exception as exc:  # noqa: BLE001 - surface a readable error to the agent
            self.log.warning("fetch_failed", url=url, error=str(exc))
            return {"url": url, "error": f"fetch failed: {exc}"}
        return result.to_dict()
