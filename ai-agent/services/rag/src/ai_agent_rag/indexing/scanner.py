"""Directory scanning + per-file fingerprinting for change detection."""

from __future__ import annotations

import fnmatch
import hashlib
from dataclasses import dataclass
from pathlib import Path

from ..config import DirectorySource


@dataclass
class ScannedFile:
    path: Path
    source_id: str  # stable id: "<source.id>:<relative-posix-path>"
    content_hash: str
    size: int
    mtime: float


def _hash_file(path: Path) -> str:
    digest = hashlib.blake2b(digest_size=16)
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _excluded(rel_posix: str, patterns: list[str]) -> bool:
    # fnmatch treats '*' as matching across '/', so "drafts/**" and "**/drafts/**" both work here.
    return any(fnmatch.fnmatch(rel_posix, pattern) for pattern in patterns)


def scan_directory(
    source: DirectorySource, *, supported_extensions: set[str] | None = None
) -> list[ScannedFile]:
    """Discover files under a directory source matching include/exclude globs.

    When ``supported_extensions`` is given, files with other extensions are skipped (and logged upstream).
    """
    root = Path(source.path)
    if not root.exists():
        return []

    discovered: dict[Path, ScannedFile] = {}
    for pattern in source.include:
        for path in root.glob(pattern):
            if not path.is_file():
                continue
            # Honor `recursive: false` regardless of the include glob (e.g. the default "**/*").
            if not source.recursive and path.parent != root:
                continue
            if supported_extensions is not None and path.suffix.lower() not in supported_extensions:
                continue
            rel = path.relative_to(root)
            if _excluded(rel.as_posix(), source.exclude):
                continue
            if path in discovered:
                continue
            stat = path.stat()
            discovered[path] = ScannedFile(
                path=path,
                source_id=f"{source.id}:{rel.as_posix()}",
                content_hash=_hash_file(path),
                size=stat.st_size,
                mtime=stat.st_mtime,
            )
    return list(discovered.values())
