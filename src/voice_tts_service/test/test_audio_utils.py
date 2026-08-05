import io
import math
import struct
import wave

import pytest

from voice_tts_service.audio_utils import decode_wav, float_pcm_to_wav, to_mono
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
