"""Document loader protocol + an extension→loader registry."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from ..models import ParsedDocument
from .native import HtmlLoader, MarkdownLoader, TextLoader


@runtime_checkable
class DocumentLoader(Protocol):
    def load(self, path: Path, source_id: str) -> ParsedDocument: ...


class LoaderRegistry:
    def __init__(self) -> None:
        self._by_ext: dict[str, DocumentLoader] = {}

    def register(self, extensions: list[str], loader: DocumentLoader) -> None:
        for ext in extensions:
            self._by_ext[ext.lower()] = loader

    def for_path(self, path: Path) -> DocumentLoader | None:
        return self._by_ext.get(path.suffix.lower())

    def supported_extensions(self) -> set[str]:
        return set(self._by_ext)


def build_default_registry(use_docling: bool = False) -> LoaderRegistry:
    """Native loaders for text/markdown/html; Docling (if enabled) for PDF/Office/images."""
    registry = LoaderRegistry()
    registry.register([".txt", ".text", ".log", ".csv"], TextLoader())
    registry.register([".md", ".markdown"], MarkdownLoader())
    registry.register([".html", ".htm"], HtmlLoader())

    if use_docling:
        from .docling_loader import DoclingLoader

        docling = DoclingLoader()
        registry.register(
            [".pdf", ".docx", ".pptx", ".xlsx", ".png", ".jpg", ".jpeg", ".tiff"], docling
        )
    return registry
