"""Tests for Reciprocal Rank Fusion + hit merging."""

from __future__ import annotations

from ai_agent_rag.fusion import fuse, reciprocal_rank_fusion
from ai_agent_rag.models import RetrievedHit


def _hit(cid: str, text: str = "") -> RetrievedHit:
    return RetrievedHit(text=text or cid, score=0.0, chunk_id=cid, source_id="s", metadata={})


def test_rrf_rewards_agreement_across_rankings() -> None:
    scores = reciprocal_rank_fusion([["a", "b"], ["b", "c"]], k=60)
    # b appears in both lists -> highest; a (rank 1 in one) > c (rank 2 in one)
    assert scores["b"] > scores["a"] > scores["c"]


def test_fuse_merges_and_ranks_by_rrf() -> None:
    dense = [_hit("a"), _hit("b")]
    sparse = [_hit("b"), _hit("c")]
    fused = fuse(dense, sparse, top_k=3)
    assert [h.chunk_id for h in fused] == ["b", "a", "c"]
    assert fused[0].score > fused[1].score  # score is the RRF score


def test_fuse_prefers_dense_hit_text_on_overlap() -> None:
    dense = [_hit("b", text="dense-text")]
    sparse = [_hit("b", text="sparse-text")]
    fused = fuse(dense, sparse, top_k=1)
    assert fused[0].text == "dense-text"


def test_fuse_respects_top_k() -> None:
    dense = [_hit("a"), _hit("b"), _hit("c")]
    assert len(fuse(dense, [], top_k=2)) == 2
