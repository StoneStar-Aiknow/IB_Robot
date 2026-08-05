import pytest

from voice_tts_service.backend import AudioResult
from voice_tts_service.errors import TTSError
from voice_tts_service.service_core import TTSLimits, TTSServiceCore


class FakeBackend:
    runtime_version = "fake-1"

    def __init__(self):
        self.calls = []

    def load(self):
        return None

    def synthesize(self, text, prompt_audio, prompt_text):
        self.calls.append((text, prompt_audio, prompt_text))
        return AudioResult(samples=[0.0] * 16, sample_rate=8000)

    def close(self):
        return None


def test_fake_backend_produces_ordered_independent_wav_segments():
    backend = FakeBackend()
    core = TTSServiceCore(
        backend,
        TTSLimits(segment_max_chars=5, segment_pause_ms=120, max_request_chars=20, max_segments=4),
    )

    output = core.synthesize("abcdefgh")

    assert [segment.text for segment in output.segments] == ["abcde", "fgh"]
    assert [segment.pause_after_ms for segment in output.segments] == [120, 0]
    assert all(segment.wav_data.startswith(b"RIFF") for segment in output.segments)
    assert [call[0] for call in backend.calls] == ["abcde", "fgh"]


def test_not_ready_request_and_response_limit_are_explicit():
    with pytest.raises(TTSError) as not_ready:
        TTSServiceCore(None, TTSLimits()).synthesize("hello")
    assert not_ready.value.code == "MODEL_NOT_READY"

    core = TTSServiceCore(FakeBackend(), TTSLimits(max_response_audio_bytes=10))
    with pytest.raises(TTSError) as too_large:
        core.synthesize("hello")
    assert too_large.value.code == "RESPONSE_TOO_LARGE"


def test_request_limits_are_checked_without_a_loaded_backend():
    core = TTSServiceCore(None, TTSLimits(max_request_chars=4, max_prompt_audio_bytes=4))

    with pytest.raises(TTSError) as text_too_large:
        core.prepare_request("hello")
    assert text_too_large.value.code == "REQUEST_TOO_LARGE"

    class OversizedPrompt:
        def __len__(self):
            return 5

        def __bytes__(self):
            raise AssertionError("oversized prompt must be rejected before copying")

    with pytest.raises(TTSError) as prompt_too_large:
        core.prepare_request("ok", OversizedPrompt(), "wav", "prompt")
    assert prompt_too_large.value.code == "PROMPT_TOO_LARGE"
