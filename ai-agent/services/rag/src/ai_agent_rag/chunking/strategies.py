"""Chunking strategies (pure Python, dependency-free).

Token counts are approximated by whitespace words — good enough for splitting decisions and keeps the
package light. Implemented: fixed, recursive, sentence, markdown-structure. Semantic / parent-child /
late chunking are documented as future strategies in docs/subprojects/rag.md §5.
"""

from __future__ import annotations

import hashlib
import re

from ..models import Chunk, ParsedDocument

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_HEADING = re.compile(r"^#{1,6}\s+.*$", re.MULTILINE)


def _content_hash(text: str) -> str:
    return hashlib.blake2b(text.encode("utf-8"), digest_size=16).hexdigest()


def _make_chunk(doc: ParsedDocument, index: int, text: str, extra: dict | None = None) -> Chunk:
    metadata = {
        **doc.metadata,
        "path": doc.path,
        "chunk_index": index,
        "content_hash": _content_hash(text),
    }
    if extra:
        metadata.update(extra)
    return Chunk(
        id=f"{doc.source_id}#{index}",
        text=text,
        source_id=doc.source_id,
        path=doc.path,
        metadata=metadata,
    )


def _wc(text: str) -> int:
    return len(text.split())


class FixedSizeChunker:
    strategy = "fixed"

    def __init__(self, target_tokens: int = 512, overlap_tokens: int = 0) -> None:
        self.target = max(1, target_tokens)
        self.overlap = max(0, min(overlap_tokens, self.target - 1))

    def split(self, doc: ParsedDocument) -> list[Chunk]:
        words = doc.text.split()
        if not words:
            return []
        step = self.target - self.overlap
        chunks: list[Chunk] = []
        for start in range(0, len(words), step):
            window = words[start : start + self.target]
            if not window:
                break
            chunks.append(_make_chunk(doc, len(chunks), " ".join(window)))
        return chunks


def _recursive_split(text: str, target: int, separators: list[str]) -> list[str]:
    text = text.strip()
    if not text:
        return []
    if _wc(text) <= target or not separators:
        return [text]

    sep = separators[0]
    rest = separators[1:]
    if sep not in text:
        return _recursive_split(text, target, rest)

    parts = text.split(sep)
    packed: list[str] = []
    current: list[str] = []
    current_wc = 0
    for part in parts:
        pw = _wc(part)
        if current and current_wc + pw > target:
            packed.append(sep.join(current))
            current, current_wc = [part], pw
        else:
            current.append(part)
            current_wc += pw
    if current:
        packed.append(sep.join(current))

    result: list[str] = []
    for piece in packed:
        if _wc(piece) > target and rest:
            result.extend(_recursive_split(piece, target, rest))
        elif piece.strip():
            result.append(piece.strip())
    return result


class RecursiveChunker:
    strategy = "recursive"

    def __init__(self, target_tokens: int = 512) -> None:
        self.target = max(1, target_tokens)

    def split(self, doc: ParsedDocument) -> list[Chunk]:
        pieces = _recursive_split(doc.text, self.target, ["\n\n", "\n", ". ", " "])
        return [_make_chunk(doc, i, piece) for i, piece in enumerate(pieces)]


class SentenceChunker:
    strategy = "sentence"

    def __init__(self, target_tokens: int = 512) -> None:
        self.target = max(1, target_tokens)

    def split(self, doc: ParsedDocument) -> list[Chunk]:
        sentences = [s.strip() for s in _SENTENCE_SPLIT.split(doc.text) if s.strip()]
        chunks: list[Chunk] = []
        current: list[str] = []
        current_wc = 0
        for sentence in sentences:
            sw = _wc(sentence)
            if current and current_wc + sw > self.target:
                chunks.append(_make_chunk(doc, len(chunks), " ".join(current)))
                current, current_wc = [sentence], sw
            else:
                current.append(sentence)
                current_wc += sw
        if current:
            chunks.append(_make_chunk(doc, len(chunks), " ".join(current)))
        return chunks


class MarkdownChunker:
    """Structure-aware: split on headings, keeping each section together (further split if oversized)."""

    strategy = "markdown"

    def __init__(self, target_tokens: int = 1024) -> None:
        self.target = max(1, target_tokens)

    def split(self, doc: ParsedDocument) -> list[Chunk]:
        text = doc.text
        heading_positions = [m.start() for m in _HEADING.finditer(text)]
        if not heading_positions:
            return RecursiveChunker(self.target).split(doc)

        bounds = [*heading_positions, len(text)]
        sections: list[str] = []
        if heading_positions[0] > 0:
            preamble = text[: heading_positions[0]].strip()
            if preamble:
                sections.append(preamble)
        for i in range(len(heading_positions)):
            sections.append(text[bounds[i] : bounds[i + 1]].strip())

        chunks: list[Chunk] = []
        for section in sections:
            if not section:
                continue
            title = (
                section.splitlines()[0].lstrip("# ").strip() if section.startswith("#") else None
            )
            pieces = (
                [section]
                if _wc(section) <= self.target
                else _recursive_split(section, self.target, ["\n\n", "\n", ". ", " "])
            )
            for piece in pieces:
                chunks.append(
                    _make_chunk(
                        doc, len(chunks), piece, extra={"section": title} if title else None
                    )
                )
        return chunks
