"""Dependency-light loaders for text, Markdown, and HTML."""

from __future__ import annotations

import re
from pathlib import Path

from ..models import ParsedDocument

_MD_HEADING = re.compile(r"^#{1,6}\s+(.*)$", re.MULTILINE)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


class TextLoader:
    def load(self, path: Path, source_id: str) -> ParsedDocument:
        return ParsedDocument(
            source_id=source_id,
            path=str(path),
            text=_read(path),
            metadata={"mime": "text/plain", "title": path.stem},
        )


class MarkdownLoader:
    def load(self, path: Path, source_id: str) -> ParsedDocument:
        text = _read(path)
        heading = _MD_HEADING.search(text)
        title = heading.group(1).strip() if heading else path.stem
        return ParsedDocument(
            source_id=source_id,
            path=str(path),
            text=text,
            metadata={"mime": "text/markdown", "title": title},
        )


class HtmlLoader:
    def load(self, path: Path, source_id: str) -> ParsedDocument:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(_read(path), "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        title = soup.title.get_text(strip=True) if soup.title else path.stem
        body = soup.body or soup
        lines = [line.strip() for line in body.get_text(separator="\n").splitlines()]
        text = "\n".join(line for line in lines if line)
        return ParsedDocument(
            source_id=source_id,
            path=str(path),
            text=text,
            metadata={"mime": "text/html", "title": title},
        )
