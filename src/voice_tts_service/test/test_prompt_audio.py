import io
import struct
import wave
from array import array

import pytest

from voice_tts_service.errors import TTSError
from voice_tts_service.prompt_audio import decode_prompt


def _prompt_wav(frame_count=16, sample_rate=16000):
    output = io.BytesIO()
    with wave.open(output, "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        stream.writeframes(struct.pack(f"<{frame_count}h", *([0] * frame_count)))
    return output.getvalue()


def test_prompt_fields_must_be_provided_together():
    with pytest.raises(TTSError) as error:
        decode_prompt(_prompt_wav(), "", "reference", max_bytes=4096, max_duration_sec=1.0)
    assert error.value.code == "INVALID_PROMPT_PAIR"


def test_prompt_limits_and_wav_contract_are_enforced():
    data = _prompt_wav(frame_count=160)
    prompt = decode_prompt(data, "wav", "reference", max_bytes=4096, max_duration_sec=1.0)

    assert prompt is not None
    assert prompt.sample_rate == 16000
    assert prompt.duration_sec == pytest.approx(0.01)

    with pytest.raises(TTSError) as error:
        decode_prompt(data, "wav", "reference", max_bytes=16, max_duration_sec=1.0)
    assert error.value.code == "PROMPT_TOO_LARGE"


def test_prompt_decoder_accepts_ros_uint8_array_without_bytes_conversion():
    data = array("B", _prompt_wav(frame_count=160))

    prompt = decode_prompt(data, "wav", "reference", max_bytes=4096, max_duration_sec=1.0)

    assert prompt is not None
    assert prompt.duration_sec == pytest.approx(0.01)
