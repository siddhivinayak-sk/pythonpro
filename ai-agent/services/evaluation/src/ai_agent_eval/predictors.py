"""Predictors turn a golden item into a prediction to be scored.

- ``ReplayPredictor`` reads prediction fields embedded in the dataset (``pred_*``), so the shipped demo
  datasets are self-scoring and ``eval run`` produces a scored report with no live services.
- ``LiveRagPredictor`` / ``LiveAgentPredictor`` call the running RAG / chat services (lazy httpx). These
  need the services up (Docker profiles) and are not exercised by the offline test suite.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class Prediction:
    answer: str = ""
    contexts: list[str] = field(default_factory=list)
    retrieved_ids: list[str] = field(default_factory=list)
    tool_calls: list[str] = field(default_factory=list)


class Predictor(Protocol):
    def predict(self, item: dict[str, Any]) -> Prediction: ...


class ReplayPredictor:
    """Read a prediction bundled into the golden item (``pred_answer``, ``pred_contexts``, ...)."""

    def predict(self, item: dict[str, Any]) -> Prediction:
        return Prediction(
            answer=item.get("pred_answer", ""),
            contexts=list(item.get("pred_contexts", []) or []),
            retrieved_ids=list(item.get("pred_retrieved_ids", []) or []),
            tool_calls=list(item.get("pred_tool_calls", []) or []),
        )


class LiveRagPredictor:
    """Call the RAG service to retrieve contexts for a golden question (needs a live service)."""

    def __init__(self, rag_base_url: str, collection: str, k: int = 5) -> None:
        self.rag_base_url = rag_base_url.rstrip("/")
        self.collection = collection
        self.k = k

    def predict(self, item: dict[str, Any]) -> Prediction:
        import httpx

        resp = httpx.post(
            f"{self.rag_base_url}/v1/retrieve",
            json={"collection": self.collection, "query": item.get("question", ""), "k": self.k},
            timeout=30,
        )
        resp.raise_for_status()
        hits = resp.json().get("hits", [])
        return Prediction(
            contexts=[h.get("text", "") for h in hits],
            retrieved_ids=[
                (h.get("citation") or {}).get("path") or h.get("chunk_id", "") for h in hits
            ],
        )


class LiveAgentPredictor:
    """Drive the chat agent for an agent scenario (needs a live chat service)."""

    def __init__(self, chat_base_url: str, token: str) -> None:
        self.chat_base_url = chat_base_url.rstrip("/")
        self.token = token

    def predict(self, item: dict[str, Any]) -> Prediction:
        import httpx

        headers = {"Authorization": f"Bearer {self.token}"}
        with httpx.Client(base_url=self.chat_base_url, headers=headers, timeout=60) as client:
            conv = client.post("/v1/conversations", json={}).json()
            last = ""
            tools: list[str] = []
            for turn in item.get("scenario", []):
                if turn.get("role") == "user":
                    msg = client.post(
                        f"/v1/conversations/{conv['id']}/chat", json={"text": turn["content"]}
                    ).json()
                    last = msg.get("content", "")
                    tools = list(msg.get("tools_used", []))
            return Prediction(answer=last, tool_calls=tools)
