import io
import math
import struct
import wave

import numpy as np
import pytest

from voice_tts_service.audio_utils import decode_wav, float_pcm_to_wav, resample_pcm16, to_mono
from voice_tts_service.errors import TTSError


def _wav_bytes(samples, *, sample_rate=8000, channels=1):
    output = io.BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setnchannels(channels)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    return output.getvalue()


def test_float_pcm_encodes_complete_mono_pcm16_wav():
    data, duration = float_pcm_to_wav([-1.0, -0.5, 0.0, 0.5, 1.0], 1000)
    decoded = decode_wav(data)

    assert decoded.channels == 1
    assert decoded.sample_rate == 1000
    assert len(decoded.samples) == 5
    assert duration == pytest.approx(0.005)
    assert decoded.samples[0] == -1.0
    assert decoded.samples[-1] == pytest.approx(32767 / 32768)


def test_stereo_prompt_is_averaged_to_mono():
    decoded = decode_wav(_wav_bytes([32767, -32768, 16384, 16384], channels=2))

    mono = to_mono(decoded.samples, decoded.channels)

    assert len(mono) == 2
    assert mono[0] == pytest.approx(-1 / 65536)
    assert mono[1] == pytest.approx(0.5)


@pytest.mark.parametrize("samples", [[], [math.nan], [math.inf]])
def test_invalid_model_pcm_is_rejected(samples):
    with pytest.raises(TTSError) as error:
        float_pcm_to_wav(samples, 16000)
    assert error.value.code == "INVALID_AUDIO_OUTPUT"


def _tone_pcm(frequency, sample_rate, *, seconds=1.0, channels=1, amplitude=0.5):
    t = np.arange(int(sample_rate * seconds)) / sample_rate
    tone = np.round(amplitude * 32767 * np.sin(2 * np.pi * frequency * t)).astype("<i2")
    return np.repeat(tone, channels).tobytes()


def _rms(pcm, *, channels=1, channel=0):
    samples = np.frombuffer(pcm, dtype="<i2").reshape(-1, channels)[:, channel] / 32768.0
    trimmed = samples[samples.size // 10 : -(samples.size // 10)]
    return float(np.sqrt(np.mean(trimmed**2)))


@pytest.mark.parametrize(("source", "target"), [(24000, 16000), (16000, 24000), (48000, 16000)])
def test_resample_pcm16_preserves_duration(source, target):
    pcm = _tone_pcm(440, source, seconds=0.75)

    resampled = resample_pcm16(pcm, channels=1, source_rate=source, target_rate=target)

    assert len(resampled) // 2 == pytest.approx(0.75 * target, abs=1)


def test_resample_pcm16_same_rate_is_identity():
    pcm = _tone_pcm(440, 16000, seconds=0.1)

    assert resample_pcm16(pcm, channels=1, source_rate=16000, target_rate=16000) == pcm


def test_resample_pcm16_keeps_passband_tone_in_phase():
    resampled = resample_pcm16(_tone_pcm(1000, 24000), channels=1, source_rate=24000, target_rate=16000)
    samples = np.frombuffer(resampled, dtype="<i2") / 32768.0
    ideal = 0.5 * np.sin(2 * np.pi * 1000 * np.arange(16000) / 16000)

    # Away from the edges the filtered signal matches an ideal 16 kHz tone: no gain loss, no delay.
    assert np.max(np.abs(samples[1000:-1000] - ideal[1000:-1000])) < 2e-3


def test_resample_pcm16_suppresses_components_above_the_new_nyquist():
    source = _tone_pcm(11000, 24000)

    resampled = resample_pcm16(source, channels=1, source_rate=24000, target_rate=16000)

    # 11 kHz cannot be represented at 16 kHz; without filtering it would fold back to 5 kHz at full level.
    assert _rms(resampled) < 0.01 * _rms(source)


def test_resample_pcm16_keeps_interleaved_channels_separate():
    left = np.round(0.5 * 32767 * np.sin(2 * np.pi * 500 * np.arange(24000) / 24000)).astype("<i2")
    pcm = np.stack([left, np.zeros_like(left)], axis=1).tobytes()

    resampled = resample_pcm16(pcm, channels=2, source_rate=24000, target_rate=16000)

    assert _rms(resampled, channels=2, channel=0) == pytest.approx(0.5 / math.sqrt(2), rel=0.01)
    assert _rms(resampled, channels=2, channel=1) == 0.0


def test_resample_pcm16_rejects_partial_frames():
    with pytest.raises(ValueError, match="whole number"):
        resample_pcm16(b"\x00\x00\x00", channels=1, source_rate=24000, target_rate=16000)


def test_resample_pcm16_rejects_invalid_rates_and_channels():
    pcm = _tone_pcm(440, 16000, seconds=0.01)
    for kwargs in (
        {"channels": 0, "source_rate": 24000, "target_rate": 16000},
        {"channels": 1, "source_rate": 0, "target_rate": 16000},
        {"channels": 1, "source_rate": 24000, "target_rate": -1},
    ):
        with pytest.raises(ValueError):
            resample_pcm16(pcm, **kwargs)


def test_resample_pcm16_handles_empty_and_single_frame_input():
    assert resample_pcm16(b"", channels=1, source_rate=24000, target_rate=16000) == b""
    # One frame still has to produce a whole number of frames, not an exception.
    single = resample_pcm16(b"\x00\x40", channels=1, source_rate=24000, target_rate=16000)
    assert len(single) % 2 == 0
