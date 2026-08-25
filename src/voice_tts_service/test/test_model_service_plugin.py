from types import SimpleNamespace

import pytest

from inference_service.model_service_plugin import ModelServiceError, ModelServicePlugin
from inference_service.runtime_composition import build_model_service_runtime_dependencies
from voice_tts_service.errors import TTSError
from voice_tts_service.model_service_plugin import ZipVoiceSynthesizePlugin
from voice_tts_service.service_core import SynthesisOutput, SynthesizedSegment


def _response():
    return SimpleNamespace(
        audio_segments=[],
        error_code="",
        total_duration_sec=0.0,
        total_inference_time_ms=0.0,
    )


def test_zipvoice_plugin_implements_the_shared_typed_service_contract():
    assert issubclass(ZipVoiceSynthesizePlugin, ModelServicePlugin)
    assert ZipVoiceSynthesizePlugin.service_type == "ibrobot_msgs/srv/SynthesizeSpeech"


def test_zipvoice_session_builder_is_registered_by_manifest_identity():
    from voice_tts_service.model_session_builders import build_zipvoice_session

    dependencies = build_model_service_runtime_dependencies()
    try:
        assert (
            dependencies.registry_set.session_builder_registry.get("tensor_model", "zipvoice", "synthesize", "ascend")
            is build_zipvoice_session
        )
    finally:
        dependencies.providers.close()


def test_zipvoice_plugin_maps_domain_output_to_the_typed_response():
    plugin = ZipVoiceSynthesizePlugin.__new__(ZipVoiceSynthesizePlugin)
    segment = SynthesizedSegment(
        index=0,
        text="你好。",
        wav_data=b"RIFF",
        sample_rate=24000,
        duration_sec=0.25,
        inference_time_ms=12.0,
        pause_after_ms=0,
    )
    plugin._core = SimpleNamespace(
        synthesize=lambda *_args: SynthesisOutput(
            normalized_text="你好。",
            segments=(segment,),
            total_duration_sec=0.25,
            total_inference_time_ms=12.0,
        )
    )
    request = SimpleNamespace(text="你好。", prompt_audio=[], prompt_audio_format="", prompt_text="")
    response = _response()

    message = plugin.handle(request, response)

    assert message == "synthesized 1 audio segment(s)"
    assert response.error_code == ""
    assert response.total_duration_sec == 0.25
    assert response.audio_segments[0].audio_data.tobytes() == b"RIFF"


def test_zipvoice_plugin_preserves_stable_tts_error_codes():
    plugin = ZipVoiceSynthesizePlugin.__new__(ZipVoiceSynthesizePlugin)

    def fail(*_args):
        raise TTSError("REQUEST_TOO_LARGE", "too much text")

    plugin._core = SimpleNamespace(synthesize=fail)
    request = SimpleNamespace(text="x", prompt_audio=[], prompt_audio_format="", prompt_text="")

    with pytest.raises(ModelServiceError) as raised:
        plugin.handle(request, _response())

    assert raised.value.response_fields["error_code"] == "REQUEST_TOO_LARGE"


def test_zipvoice_plugin_maps_unclassified_failures_to_internal_error():
    plugin = ZipVoiceSynthesizePlugin.__new__(ZipVoiceSynthesizePlugin)

    def fail(*_args):
        raise RuntimeError("unexpected")

    plugin._core = SimpleNamespace(synthesize=fail)
    request = SimpleNamespace(text="x", prompt_audio=[], prompt_audio_format="", prompt_text="")

    with pytest.raises(ModelServiceError) as raised:
        plugin.handle(request, _response())

    assert raised.value.response_fields["error_code"] == "INTERNAL_ERROR"
