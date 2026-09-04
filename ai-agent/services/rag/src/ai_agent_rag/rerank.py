"""Cross-encoder reranking via a pluggable ``Reranker``.

The default ``NoopReranker`` preserves the input order (reranking is opt-in). ``CrossEncoderReranker``
scores each (query, chunk) pair with a sentence-transformers CrossEncoder (lazy import; needs the
``embeddings-hf`` extra) and reorders. Injectable so tests use a fake with no model download.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .models import RetrievedHit


@runtime_checkable
class Reranker(Protocol):
    def rerank(self, query: str, hits: list[RetrievedHit], top_k: int) -> list[RetrievedHit]: ...


class NoopReranker:
    """Default: keep the input ranking, just truncate to top_k."""

    def rerank(self, query: str, hits: list[RetrievedHit], top_k: int) -> list[RetrievedHit]:
        return hits[:top_k]


class CrossEncoderReranker:
    def __init__(self, model: str, *, device: str = "cpu") -> None:
        from sentence_transformers import CrossEncoder  # lazy

        self.model_name = model
        self._model = CrossEncoder(model, device=device)

    def rerank(self, query: str, hits: list[RetrievedHit], top_k: int) -> list[RetrievedHit]:
        if not hits:
            return []
        scores = self._model.predict([(query, h.text) for h in hits])
        ranked = sorted(zip(hits, scores, strict=False), key=lambda p: float(p[1]), reverse=True)
        return [
            RetrievedHit(
                text=h.text,
                score=float(score),
                chunk_id=h.chunk_id,
                source_id=h.source_id,
                metadata=h.metadata,
            )
            for h, score in ranked[:top_k]
        ]


def build_reranker(model: str | None, *, device: str = "cpu") -> Reranker:
    """Construct the configured reranker (Noop unless a cross-encoder model is set)."""
    if not model:
        return NoopReranker()
    return CrossEncoderReranker(model, device=device)
