"""Audio (STT/TTS) via a pluggable ``SpeechProvider``.

The default ``NullSpeechProvider`` raises ``SpeechError`` (surfaced as 501) so the endpoints exist but are
inert until a real provider is configured. Real providers (OpenAI, local Whisper, etc.) implement the same
two methods and are selected by config — kept out of the default install so no speech model is required.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .config import ChatSettings


class SpeechError(Exception):
    """Raised when speech is unavailable or fails."""


@runtime_checkable
class SpeechProvider(Protocol):
    def transcribe(self, audio: bytes, mime: str) -> str:
        """Speech-to-text: return the transcript of the audio."""
        ...

    def synthesize(self, text: str, voice: str) -> tuple[bytes, str]:
        """Text-to-speech: return (audio_bytes, mime_type)."""
        ...


class NullSpeechProvider:
    """No-op provider used when audio is disabled/unconfigured."""

    def transcribe(self, audio: bytes, mime: str) -> str:
        raise SpeechError("audio is not configured on this server")

    def synthesize(self, text: str, voice: str) -> tuple[bytes, str]:
        raise SpeechError("audio is not configured on this server")


def build_speech_provider(settings: ChatSettings) -> SpeechProvider:
    """Construct the configured speech provider (Null unless audio is enabled and a provider is wired)."""
    if not settings.audio_enabled:
        return NullSpeechProvider()
    # Real STT/TTS providers plug in here (lazy import by settings.stt_provider / tts_provider).
    # Until one is wired, fall back to Null so endpoints respond 501 rather than 500.
    return NullSpeechProvider()
