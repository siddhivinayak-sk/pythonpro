"""Reciprocal Rank Fusion (RRF) for combining dense + sparse result lists.

RRF is rank-based (score-scale agnostic), which is exactly what we want when merging cosine similarities
with BM25 scores: ``score(d) = sum_r 1 / (k + rank_r(d))`` over each input ranking.
"""

from __future__ import annotations

from collections.abc import Sequence

from .models import RetrievedHit

_RRF_K = 60


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[str]], *, k: int = _RRF_K
) -> dict[str, float]:
    """Return a {doc_id: fused_score} map from several ranked id lists."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


def fuse(
    dense: Sequence[RetrievedHit],
    sparse: Sequence[RetrievedHit],
    *,
    top_k: int,
    rrf_k: int = _RRF_K,
) -> list[RetrievedHit]:
    """Fuse dense + sparse hits with RRF and return the top-k (hit.score becomes the RRF score)."""
    by_id: dict[str, RetrievedHit] = {}
    for hit in [*dense, *sparse]:  # dense first -> its text/metadata wins on ties
        by_id.setdefault(hit.chunk_id, hit)
    scores = reciprocal_rank_fusion(
        [[h.chunk_id for h in dense], [h.chunk_id for h in sparse]], k=rrf_k
    )
    ordered = sorted(by_id.values(), key=lambda h: scores.get(h.chunk_id, 0.0), reverse=True)
    return [
        RetrievedHit(
            text=h.text,
            score=scores.get(h.chunk_id, 0.0),
            chunk_id=h.chunk_id,
            source_id=h.source_id,
            metadata=h.metadata,
        )
        for h in ordered[:top_k]
    ]
