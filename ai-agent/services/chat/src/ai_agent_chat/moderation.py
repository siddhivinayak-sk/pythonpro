"""Input/output moderation via a pluggable ``ModerationProvider``.

The default ``AllowAllModerationProvider`` flags nothing, so behaviour is unchanged unless moderation is
enabled. The dep-free ``KeywordModerationProvider`` flags text matching any configured (case-insensitive)
regex/substring pattern — enough for a real, offline-testable default. API-based providers (OpenAI or
Azure content-safety, a local classifier, ...) implement the same protocol and plug in lazily by name.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from ai_agent_core import get_logger

from .config import ChatSettings

log = get_logger("moderation")


@dataclass
class ModerationResult:
    flagged: bool
    categories: list[str] = field(default_factory=list)
    reason: str = ""


@runtime_checkable
class ModerationProvider(Protocol):
    def check(self, text: str, *, kind: str = "output") -> ModerationResult:
        """Return whether ``text`` violates policy. ``kind`` is ``"input"`` or ``"output"``."""
        ...


class AllowAllModerationProvider:
    """Default provider: never flags. Moderation is opt-in."""

    def check(self, text: str, *, kind: str = "output") -> ModerationResult:
        return ModerationResult(flagged=False)


class KeywordModerationProvider:
    """Rule-based moderation: flags text matching any configured pattern (case-insensitive)."""

    def __init__(self, blocklist: list[str]) -> None:
        self._patterns: list[tuple[str, re.Pattern[str]]] = []
        for term in blocklist:
            if not term:
                continue
            try:
                self._patterns.append((term, re.compile(term, re.IGNORECASE)))
            except re.error:
                # Fall back to a literal match if the entry is not a valid regex.
                self._patterns.append((term, re.compile(re.escape(term), re.IGNORECASE)))

    def check(self, text: str, *, kind: str = "output") -> ModerationResult:
        hits = sorted({term for term, pattern in self._patterns if pattern.search(text or "")})
        if hits:
            return ModerationResult(
                flagged=True, categories=["blocklist"], reason="matched: " + ", ".join(hits)
            )
        return ModerationResult(flagged=False)


def build_moderation_provider(settings: ChatSettings) -> ModerationProvider:
    """Construct the configured moderation provider (AllowAll unless moderation is enabled)."""
    if not settings.moderation_enabled:
        return AllowAllModerationProvider()
    # Real API-based providers plug in here (lazy import by a settings-selected provider name).
    return KeywordModerationProvider(settings.moderation_blocklist)
