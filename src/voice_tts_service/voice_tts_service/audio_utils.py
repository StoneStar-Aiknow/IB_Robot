"""Decode prompt WAV files and encode synthesized float PCM in memory."""

from __future__ import annotations

import io
import math
import struct
import wave
from collections.abc import Iterable
from dataclasses import dataclass

from voice_tts_service.errors import TTSError


@dataclass(frozen=True)
class DecodedWav:
    """Decoded interleaved float PCM in the WAV file's original channel layout."""

    samples: tuple[float, ...]
    sample_rate: int
    channels: int

    @property
    def duration_sec(self) -> float:
        return len(self.samples) / (self.sample_rate * self.channels)


def _decode_pcm(raw: bytes, sample_width: int) -> tuple[float, ...]:
    if sample_width == 1:
        return tuple((value - 128) / 128.0 for value in raw)
    if sample_width == 2:
        count = len(raw) // 2
        return tuple(value / 32768.0 for value in struct.unpack(f"<{count}h", raw))
    if sample_width == 3:
        values = []
        for offset in range(0, len(raw), 3):
            value = int.from_bytes(raw[offset : offset + 3], "little", signed=False)
            if value & 0x800000:
                value -= 1 << 24
            values.append(value / 8388608.0)
        return tuple(values)
    if sample_width == 4:
        count = len(raw) // 4
        return tuple(value / 2147483648.0 for value in struct.unpack(f"<{count}i", raw))
    raise TTSError("INVALID_PROMPT_AUDIO", f"unsupported WAV PCM sample width: {sample_width} bytes")


def decode_wav(data: bytes) -> DecodedWav:
    """Decode an uncompressed integer PCM WAV file without filesystem access."""

    if not data:
        raise TTSError("INVALID_PROMPT_AUDIO", "prompt WAV data is empty")
    try:
        with wave.open(io.BytesIO(data), "rb") as stream:
            channels = stream.getnchannels()
            sample_rate = stream.getframerate()
            sample_width = stream.getsampwidth()
            frame_count = stream.getnframes()
            compression = stream.getcomptype()
            raw = stream.readframes(frame_count)
    except (EOFError, OSError, wave.Error) as exc:
        raise TTSError("INVALID_PROMPT_AUDIO", f"prompt audio is not a valid WAV file: {exc}") from exc

    if compression != "NONE":
        raise TTSError("INVALID_PROMPT_AUDIO", f"compressed WAV is unsupported: {compression}")
    if channels <= 0 or sample_rate <= 0 or frame_count <= 0:
        raise TTSError("INVALID_PROMPT_AUDIO", "prompt WAV has invalid channel, rate, or frame metadata")
    if len(raw) != frame_count * channels * sample_width:
        raise TTSError("INVALID_PROMPT_AUDIO", "prompt WAV PCM payload is truncated")
    return DecodedWav(samples=_decode_pcm(raw, sample_width), sample_rate=sample_rate, channels=channels)


def to_mono(samples: tuple[float, ...], channels: int) -> tuple[float, ...]:
    """Average interleaved channels without changing amplitude semantics."""

    if channels == 1:
        return samples
    return tuple(sum(samples[offset : offset + channels]) / channels for offset in range(0, len(samples), channels))


def resample_linear(samples: tuple[float, ...], source_rate: int, target_rate: int) -> tuple[float, ...]:
    """Apply deterministic linear resampling for prompt adapter compatibility."""

    if source_rate <= 0 or target_rate <= 0:
        raise ValueError("sample rates must be positive")
    if source_rate == target_rate or len(samples) <= 1:
        return samples
    output_count = max(1, round(len(samples) * target_rate / source_rate))
    scale = source_rate / target_rate
    result = []
    for output_index in range(output_count):
        position = min(output_index * scale, len(samples) - 1)
        left = int(position)
        right = min(left + 1, len(samples) - 1)
        fraction = position - left
        result.append(samples[left] * (1.0 - fraction) + samples[right] * fraction)
    return tuple(result)


def validate_float_pcm(samples: Iterable[float], sample_rate: int) -> tuple[float, ...]:
    """Materialize and validate mono logical PCM before WAV encoding."""

    if sample_rate <= 0:
        raise TTSError("INVALID_AUDIO_OUTPUT", f"invalid output sample rate: {sample_rate}")
    materialized = tuple(float(value) for value in samples)
    if not materialized:
        raise TTSError("INVALID_AUDIO_OUTPUT", "model returned empty audio")
    if any(not math.isfinite(value) for value in materialized):
        raise TTSError("INVALID_AUDIO_OUTPUT", "model audio contains NaN or infinity")
    return materialized


def float_pcm_to_wav(samples: Iterable[float], sample_rate: int) -> tuple[bytes, float]:
    """Encode mono float PCM as a complete WAV PCM16 file."""

    values = validate_float_pcm(samples, sample_rate)
    pcm = bytearray()
    for value in values:
        clipped = max(-1.0, min(1.0, value))
        integer = -32768 if clipped <= -1.0 else round(clipped * 32767.0)
        pcm.extend(struct.pack("<h", integer))

    output = io.BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(bytes(pcm))
    return output.getvalue(), len(values) / sample_rate
