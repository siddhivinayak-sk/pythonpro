"""Tests for loaders and the directory scanner."""

from __future__ import annotations

from pathlib import Path

from ai_agent_rag.config import DirectorySource
from ai_agent_rag.indexing import scan_directory
from ai_agent_rag.loaders import HtmlLoader, MarkdownLoader, TextLoader, build_default_registry


def test_text_loader(tmp_path: Path) -> None:
    p = tmp_path / "a.txt"
    p.write_text("hello world", encoding="utf-8")
    doc = TextLoader().load(p, "src:a.txt")
    assert doc.text == "hello world"
    assert doc.metadata["mime"] == "text/plain"


def test_markdown_loader_title_from_heading(tmp_path: Path) -> None:
    p = tmp_path / "b.md"
    p.write_text("# My Title\n\nbody text", encoding="utf-8")
    doc = MarkdownLoader().load(p, "src:b.md")
    assert doc.metadata["title"] == "My Title"


def test_html_loader_strips_tags(tmp_path: Path) -> None:
    p = tmp_path / "c.html"
    p.write_text(
        "<html><head><title>Doc</title></head><body><script>x()</script><p>Visible.</p></body></html>",
        encoding="utf-8",
    )
    doc = HtmlLoader().load(p, "src:c.html")
    assert "Visible." in doc.text
    assert "x()" not in doc.text
    assert doc.metadata["title"] == "Doc"


def test_registry_extension_mapping() -> None:
    registry = build_default_registry()
    exts = registry.supported_extensions()
    assert {".txt", ".md", ".html"} <= exts
    assert registry.for_path(Path("x.md")) is not None
    assert registry.for_path(Path("x.unknown")) is None


def test_scan_directory_include_exclude_recursive(tmp_path: Path) -> None:
    (tmp_path / "keep.md").write_text("# a", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("b", encoding="utf-8")
    (tmp_path / "image.bin").write_text("c", encoding="utf-8")  # unsupported
    drafts = tmp_path / "drafts"
    drafts.mkdir()
    (drafts / "wip.md").write_text("# draft", encoding="utf-8")

    source = DirectorySource(
        id="docs", path=str(tmp_path), recursive=True, include=["**/*"], exclude=["drafts/**"]
    )
    registry = build_default_registry()
    found = scan_directory(source, supported_extensions=registry.supported_extensions())
    ids = {f.source_id for f in found}

    assert "docs:keep.md" in ids
    assert "docs:notes.txt" in ids
    assert not any("image.bin" in i for i in ids)  # unsupported extension skipped
    assert not any("drafts" in i for i in ids)  # excluded
    # content hash is populated and stable
    assert all(f.content_hash for f in found)
