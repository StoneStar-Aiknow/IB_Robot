"""Config rules for the optional per-utterance voiceprint producer and its consumer."""

from __future__ import annotations

import pytest

from robot_config.loader import validate_speaker_embedding_config


def _config(speaker_embedding=None, *, voice_asr_enabled=True, voice_bridge=None) -> dict:
    voice_asr: dict = {"enabled": voice_asr_enabled, "input_mode": "external_vad"}
    if speaker_embedding is not None:
        voice_asr["speaker_embedding"] = speaker_embedding
    config: dict = {"voice_asr": voice_asr}
    if voice_bridge is not None:
        config["embodied"] = {"agent": {"voice_bridge": voice_bridge}}
    return config


def test_a_config_without_the_producer_is_valid():
    assert validate_speaker_embedding_config(_config()) == []
    assert validate_speaker_embedding_config({}) == []


@pytest.mark.parametrize(
    ("speaker_embedding", "fragment"),
    [
        pytest.param("on", "must be a mapping", id="not a mapping"),
        pytest.param({"enabled": "yes"}, "enabled must be a boolean", id="enabled not a bool"),
        pytest.param({"enabled": True}, "bundle_path is required", id="no bundle"),
        pytest.param(
            {"enabled": True, "bundle_path": "models/campplus-speaker"},
            "deployment is required",
            id="no deployment",
        ),
        pytest.param(
            {
                "enabled": True,
                "bundle_path": "models/campplus-speaker",
                "deployment": "torch_cpu",
                "output_topic": "voice/prints",
            },
            "output_topic must be an absolute ROS topic name",
            id="relative topic",
        ),
        pytest.param(
            {
                "enabled": True,
                "bundle_path": "models/campplus-speaker",
                "deployment": "torch_cpu",
                "min_speech_seconds": 0,
            },
            "min_speech_seconds must be finite and positive",
            id="zero floor",
        ),
        pytest.param(
            {
                "enabled": True,
                "bundle_path": "models/campplus-speaker",
                "deployment": "torch_cpu",
                "max_speech_seconds": -1,
            },
            "max_speech_seconds must be finite and positive",
            id="negative ceiling",
        ),
        pytest.param(
            {
                "enabled": True,
                "bundle_path": "models/campplus-speaker",
                "deployment": "torch_cpu",
                "min_speech_seconds": 3.0,
                "max_speech_seconds": 1.0,
            },
            "max_speech_seconds must not be below min_speech_seconds",
            id="ceiling below floor",
        ),
        pytest.param(
            {
                "enabled": True,
                "bundle_path": "models/campplus-speaker",
                "deployment": "torch_cpu",
                "idle_timeout_sec": 0,
            },
            "idle_timeout_sec must be finite and positive",
            id="zero idle timeout",
        ),
        pytest.param(
            {
                "enabled": True,
                "bundle_path": "models/campplus-speaker",
                "deployment": "torch_cpu",
                "exit_on_init_failure": "no",
            },
            "exit_on_init_failure must be a boolean",
            id="exit flag not a bool",
        ),
    ],
)
def test_the_producer_shape_is_validated(speaker_embedding, fragment):
    errors = validate_speaker_embedding_config(_config(speaker_embedding))

    assert any(fragment in error for error in errors)


def test_the_producer_needs_the_voice_asr_it_rides_on():
    config = _config(
        {"enabled": True, "bundle_path": "models/campplus-speaker", "deployment": "torch_cpu"}, voice_asr_enabled=False
    )

    errors = validate_speaker_embedding_config(config)

    assert any("requires voice_asr.enabled=true" in error for error in errors)


def test_a_voiceprint_check_without_a_producer_is_rejected():
    """It would read as a filter while accepting every utterance, which review cannot see."""
    config = _config(voice_bridge={"speaker_check_enabled": True, "wake_phrases": ["灵犀灵犀"]})

    errors = validate_speaker_embedding_config(config)

    assert any("nothing would publish the voiceprints to check against" in error for error in errors)


def test_a_producer_without_a_consumer_is_valid():
    """Publishing voiceprints nobody checks is a diagnostic, not a misconfiguration."""
    config = _config(
        {"enabled": True, "bundle_path": "models/campplus-speaker", "deployment": "torch_cpu"},
        voice_bridge={"enabled": True},
    )

    assert validate_speaker_embedding_config(config) == []


def test_the_voiceprint_must_finalize_before_the_transcript():
    """Equal idle timeouts make both producers fire on the same stall, and when
    the transcript wins the race the check sees no measurement and fails open.
    Observed on the X2 with both at 2.0 s: an intruder went through unchecked."""
    config = _config(
        {"enabled": True, "bundle_path": "models/campplus-speaker", "deployment": "torch_cpu", "idle_timeout_sec": 2.0},
        voice_bridge={"speaker_check_enabled": True, "addressee_trigger": "vendor_session"},
    )
    config["voice_asr"]["external_idle_timeout"] = 2.0

    errors = validate_speaker_embedding_config(config)

    assert any("must be below" in error and "external_idle_timeout" in error for error in errors)


def test_the_default_timeouts_already_finalize_the_voiceprint_first():
    """Turning the check on must not require tuning two timeouts by hand."""
    config = _config(
        {"enabled": True, "bundle_path": "models/campplus-speaker", "deployment": "torch_cpu"},
        voice_bridge={"speaker_check_enabled": True, "addressee_trigger": "vendor_session"},
    )

    assert validate_speaker_embedding_config(config) == []


def test_a_voiceprint_that_finalizes_first_is_accepted():
    config = _config(
        {"enabled": True, "bundle_path": "models/campplus-speaker", "deployment": "torch_cpu", "idle_timeout_sec": 1.2},
        voice_bridge={"speaker_check_enabled": True, "addressee_trigger": "vendor_session"},
    )
    config["voice_asr"]["external_idle_timeout"] = 2.0

    assert validate_speaker_embedding_config(config) == []
