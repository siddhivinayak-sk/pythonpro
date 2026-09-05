"""RAG integration: an HTTP client to the RAG service and a retrieval tool for the agent."""

from __future__ import annotations

from typing import Any, Protocol

from .tools import ToolResult


class RetrievalClient(Protocol):
    def retrieve(self, collection: str, query: str, k: int) -> list[dict[str, Any]]: ...


class RagClient:
    """Thin client over the RAG service's ``/v1/retrieve`` and ``/v1/collections`` endpoints."""

    def __init__(self, base_url: str, timeout: int = 30) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def retrieve(self, collection: str, query: str, k: int) -> list[dict[str, Any]]:
        import httpx

        resp = httpx.post(
            f"{self.base_url}/v1/retrieve",
            json={"collection": collection, "query": query, "k": k},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json().get("hits", [])

    def list_collections(self) -> list[str]:
        """Return the RAG service's collection names (empty list if unreachable)."""
        import httpx

        try:
            resp = httpx.get(f"{self.base_url}/v1/collections", timeout=self.timeout)
            resp.raise_for_status()
            return [c["name"] for c in resp.json().get("collections", []) if c.get("name")]
        except Exception:  # noqa: BLE001 - discovery is best-effort; never break the UI
            return []


class RagRetrievalTool:
    """Retrieve grounding passages from the selected knowledge-base collections."""

    name = "retrieve"
    description = (
        "Search the knowledge base for passages relevant to the user's question. "
        "Use this whenever the answer may be in the indexed documents; cite what you use."
    )

    def __init__(self, client: RetrievalClient, collections: list[str], k: int = 5) -> None:
        self._client = client
        self._collections = collections
        self._k = k

    def json_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "the search query"}},
            "required": ["query"],
        }

    def run(self, **kwargs: Any) -> ToolResult:
        query = str(kwargs.get("query", ""))
        hits: list[dict[str, Any]] = []
        for collection in self._collections:
            try:
                hits.extend(self._client.retrieve(collection, query, self._k))
            except Exception:  # noqa: BLE001 - a bad collection shouldn't sink the whole tool
                continue
        hits.sort(key=lambda h: h.get("score", 0.0), reverse=True)
        hits = hits[: self._k]
        if not hits:
            return ToolResult(text="No relevant passages found.")
        text = "\n\n".join(f"[{i + 1}] {h.get('text', '')}" for i, h in enumerate(hits))
        citations = [
            {
                "path": (h.get("citation") or {}).get("path"),
                "title": (h.get("citation") or {}).get("title"),
                "score": h.get("score"),
                "source": "rag",
            }
            for h in hits
        ]
        return ToolResult(text=text, citations=citations)
