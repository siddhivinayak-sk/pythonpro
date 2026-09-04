"""Chunker protocol + factory."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from ..config import ChunkerProfile
from ..models import Chunk, ParsedDocument
from .strategies import FixedSizeChunker, MarkdownChunker, RecursiveChunker, SentenceChunker


@runtime_checkable
class Chunker(Protocol):
    def split(self, doc: ParsedDocument) -> list[Chunk]: ...


def build_chunker(profile: ChunkerProfile) -> Chunker:
    if profile.strategy == "fixed":
        return FixedSizeChunker(profile.target_tokens, profile.overlap_tokens)
    if profile.strategy == "recursive":
        return RecursiveChunker(profile.target_tokens)
    if profile.strategy == "sentence":
        return SentenceChunker(profile.target_tokens)
    if profile.strategy == "markdown":
        return MarkdownChunker(profile.max_tokens or profile.target_tokens)
    raise ValueError(f"unknown chunking strategy '{profile.strategy}'")
