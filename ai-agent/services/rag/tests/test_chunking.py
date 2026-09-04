"""Tests for chunking strategies."""

from __future__ import annotations

import pytest
from ai_agent_rag.chunking import (
    FixedSizeChunker,
    MarkdownChunker,
    RecursiveChunker,
    SentenceChunker,
    build_chunker,
)
from ai_agent_rag.config import ChunkerProfile
from ai_agent_rag.models import ParsedDocument


def _doc(text: str) -> ParsedDocument:
    return ParsedDocument(source_id="s1", path="/docs/x.txt", text=text, metadata={"title": "X"})


def test_fixed_size_chunker_windows_and_metadata() -> None:
    doc = _doc(" ".join(f"w{i}" for i in range(12)))
    chunks = FixedSizeChunker(target_tokens=5, overlap_tokens=0).split(doc)
    assert len(chunks) == 3  # 12 words / 5
    assert chunks[0].id == "s1#0"
    assert chunks[0].metadata["chunk_index"] == 0
    assert chunks[0].metadata["path"] == "/docs/x.txt"
    assert chunks[0].metadata["content_hash"]
    assert len(chunks[0].text.split()) == 5


def test_fixed_size_overlap() -> None:
    doc = _doc(" ".join(f"w{i}" for i in range(10)))
    chunks = FixedSizeChunker(target_tokens=5, overlap_tokens=2).split(doc)
    # step = 3 -> starts 0,3,6,9
    assert len(chunks) == 4


def test_recursive_chunker_respects_target() -> None:
    doc = _doc("\n\n".join("sentence here " * 20 for _ in range(3)))
    chunks = RecursiveChunker(target_tokens=40).split(doc)
    assert len(chunks) >= 3
    assert all(len(c.text.split()) <= 60 for c in chunks)  # approx cap with slack


def test_sentence_chunker_groups_sentences() -> None:
    doc = _doc("One two three. Four five six. Seven eight nine. Ten eleven twelve.")
    chunks = SentenceChunker(target_tokens=6).split(doc)
    assert len(chunks) >= 2


def test_markdown_chunker_splits_on_headings() -> None:
    doc = _doc("# Title\nintro\n\n## Section A\nalpha beta\n\n## Section B\ngamma delta")
    chunks = MarkdownChunker(target_tokens=1000).split(doc)
    texts = [c.text for c in chunks]
    assert any("Section A" in t for t in texts)
    assert any("Section B" in t for t in texts)


def test_empty_text_yields_no_chunks() -> None:
    assert FixedSizeChunker().split(_doc("")) == []


def test_build_chunker_factory_and_unknown() -> None:
    assert isinstance(build_chunker(ChunkerProfile(id="c", strategy="sentence")), SentenceChunker)
    with pytest.raises(ValueError):
        build_chunker(ChunkerProfile(id="c", strategy="bogus"))
