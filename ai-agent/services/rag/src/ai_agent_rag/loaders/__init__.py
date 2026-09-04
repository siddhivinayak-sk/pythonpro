"""Multi-format document loaders + registry."""

from __future__ import annotations

from .base import DocumentLoader, LoaderRegistry, build_default_registry
from .native import HtmlLoader, MarkdownLoader, TextLoader

__all__ = [
    "DocumentLoader",
    "LoaderRegistry",
    "build_default_registry",
    "TextLoader",
    "MarkdownLoader",
    "HtmlLoader",
]
