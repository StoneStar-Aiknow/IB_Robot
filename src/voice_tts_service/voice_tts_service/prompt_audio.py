"""Request-scoped prompt audio validation and decoding."""

from __future__ import annotations

from dataclasses import dataclass

from voice_tts_service.audio_utils import decode_wav, resample_linear, to_mono
from voice_tts_service.errors import TTSError


@dataclass(frozen=True)
class PromptAudio:
    """Decoded mono prompt audio with its explicit sample-rate time base."""

    samples: tuple[float, ...]
    sample_rate: int

    @property
    def duration_sec(self) -> float:
        return len(self.samples) / self.sample_rate

    def resampled(self, sample_rate: int) -> PromptAudio:
        return PromptAudio(resample_linear(self.samples, self.sample_rate, sample_rate), sample_rate)


def validate_prompt_pair(audio: bytes, audio_format: str, text: str) -> bool:
    """Validate optional prompt fields and return whether a request prompt is present."""

    has_audio = bool(audio)
    has_text = bool(text.strip())
    has_format = bool(audio_format.strip())
    if has_audio != has_text or has_audio != has_format:
        raise TTSError(
            "INVALID_PROMPT_PAIR",
            "prompt_audio, prompt_audio_format, and prompt_text must be provided together",
        )
    return has_audio


def decode_prompt(
    audio: bytes,
    audio_format: str,
    text: str,
    *,
    max_bytes: int,
    max_duration_sec: float,
) -> PromptAudio | None:
    """Validate and decode the optional request prompt without model-specific normalization."""

    if not validate_prompt_pair(audio, audio_format, text):
        return None
    if audio_format.strip().lower() != "wav":
        raise TTSError("INVALID_PROMPT_AUDIO", "prompt_audio_format must be 'wav'")
    if len(audio) > max_bytes:
        raise TTSError("PROMPT_TOO_LARGE", f"prompt audio exceeds {max_bytes} bytes")
    decoded = decode_wav(audio)
    prompt = PromptAudio(samples=to_mono(decoded.samples, decoded.channels), sample_rate=decoded.sample_rate)
    if prompt.duration_sec > max_duration_sec:
        raise TTSError(
            "PROMPT_TOO_LARGE",
            f"prompt audio duration {prompt.duration_sec:.3f}s exceeds {max_duration_sec:.3f}s",
        )
    return prompt
