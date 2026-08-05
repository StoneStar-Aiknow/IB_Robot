"""Backend-neutral Voice TTS contracts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from voice_tts_service.prompt_audio import PromptAudio


@dataclass(frozen=True)
class AudioResult:
    """Logical mono float PCM returned by one backend synthesis call."""

    samples: Sequence[float]
    sample_rate: int


class TTSBackend(Protocol):
    """One loaded, reusable model instance."""

    runtime_version: str

    def load(self) -> None: ...

    def synthesize(self, text: str, prompt_audio: PromptAudio | None, prompt_text: str) -> AudioResult: ...

    def close(self) -> None: ...
