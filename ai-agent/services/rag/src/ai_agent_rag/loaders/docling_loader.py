"""Docling loader (default multi-format parser: PDF, Office, images w/ OCR, HTML → structured markdown).

Requires the ``ingestion`` extra (``ai-agent-rag[ingestion]``). Lazy-imported.
"""

from __future__ import annotations

from pathlib import Path

from ..models import ParsedDocument


class DoclingLoader:
    def __init__(self) -> None:
        from docling.document_converter import DocumentConverter

        self._converter = DocumentConverter()

    def load(self, path: Path, source_id: str) -> ParsedDocument:
        result = self._converter.convert(str(path))
        text = result.document.export_to_markdown()
        return ParsedDocument(
            source_id=source_id,
            path=str(path),
            text=text,
            metadata={"mime": "application/octet-stream", "title": path.stem, "parser": "docling"},
        )
