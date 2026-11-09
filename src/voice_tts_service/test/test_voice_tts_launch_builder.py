"""Launch-builder tests for the voice_tts entry (voice_tts_service's contribution).

Migrated from ``robot_config/test/test_voice_tts_launch_builder.py``
(openspec change ``migrate-business-launch-builders``, task B.5): the tests
follow the module to its owning package; assertions are verbatim, only the
import path changed.
"""

import pytest

from voice_tts_service.launch_builders import voice_tts


def _node_parameters(node):
    parameters = {}
    for raw_key, raw_value in node._Node__parameters[0].items():
        key = "".join(getattr(item, "text", str(item)) for item in raw_key)
        if isinstance(raw_value, tuple):
            raw_value = "".join(getattr(item, "text", str(item)) for item in raw_value).strip()
            if raw_value.endswith("\n..."):
                raw_value = raw_value[:-4].rstrip()
        parameters[key] = raw_value
    return parameters


def _runtime_options(node):
    """Parse runtime_options_json out of the YAML-wrapped launch parameter."""
    import json

    raw = _node_parameters(node)["runtime_options_json"].strip()
    if raw.startswith("'") and raw.endswith("'"):
        raw = raw[1:-1]
    return json.loads("".join(line.strip() for line in raw.splitlines()))


def test_voice_tts_builder_does_not_open_bundle_before_node_start(monkeypatch, tmp_path):
    defaults = {
        "enabled": False,
        "bundle_path": "models/zipvoice",
        "deployment": "",
        "exit_on_init_failure": False,
    }
    monkeypatch.setattr(voice_tts, "_load_voice_tts_service", lambda: defaults)

    nodes = voice_tts.generate_voice_tts_nodes(
        {
            "voice_tts": {
                "enabled": True,
                "bundle_path": str(tmp_path / "not-mounted-yet"),
                "deployment": "ascend_310p",
                "exit_on_init_failure": False,
            },
            "audio_io": {"enabled": True},
        }
    )

    assert len(nodes) == 2


def test_voice_tts_builder_routes_shared_playback_format(monkeypatch, tmp_path):
    defaults = {
        "enabled": False,
        "bundle_path": "models/voice_tts/zipvoice",
        "deployment": "",
        "exit_on_init_failure": False,
    }
    monkeypatch.setattr(voice_tts, "_load_voice_tts_service", lambda: defaults)

    nodes = voice_tts.generate_voice_tts_nodes(
        {
            "voice_tts": {
                "enabled": True,
                "bundle_path": str(tmp_path),
                "deployment": "ubuntu_onnx",
            },
            "audio_io": {
                "enabled": True,
                "playback_topic": "/shared/play",
                "playback_sample_rate": 24000,
                "playback_channels": 1,
            },
        }
    )

    playback = _node_parameters(nodes[1])
    assert playback["audio_topic"] == "/shared/play"
    assert playback["playback_sample_rate"] == 24000
    assert playback["playback_channels"] == 1


def test_voice_tts_builder_forwards_tts_limits_to_the_plugin(monkeypatch, tmp_path):
    defaults = {
        "enabled": False,
        "bundle_path": "models/zipvoice",
        "deployment": "",
        "device_id": 0,
        "prompt_profile": "default",
        "max_request_chars": 4000,
        "max_prompt_audio_bytes": 10 * 1024 * 1024,
        "max_prompt_duration_sec": 30.0,
        "segment_max_chars": 200,
        "segment_pause_ms": 150,
        "max_segments": 32,
        "max_response_audio_bytes": 64 * 1024 * 1024,
        "exit_on_init_failure": False,
    }
    monkeypatch.setattr(voice_tts, "_load_voice_tts_service", lambda: defaults)

    nodes = voice_tts.generate_voice_tts_nodes(
        {
            "voice_tts": {
                "enabled": True,
                "bundle_path": str(tmp_path / "zipvoice"),
                "deployment": "ascend_310p",
                "device_id": 0,
                "max_request_chars": 100,
                "max_prompt_audio_bytes": 1024,
                "max_prompt_duration_sec": 3.0,
                "exit_on_init_failure": False,
            },
            "audio_io": {"enabled": True},
        }
    )

    runtime_options = _node_parameters(nodes[0])["runtime_options_json"]
    assert '"max_request_chars": 100' in runtime_options
    assert '"max_prompt_audio_bytes": 1024' in runtime_options


# --- runtime-bound playback output (runtime-owned-audio-io) ----------------------


def _bound_audio_output(**metadata):
    return {
        "interface": "audio.play",
        "type": "audio_common_msgs/msg/AudioDataStamped",
        "topic": "/audio/play",
        "_interface_source": {"id": "audio.play"},
        **metadata,
    }


def _bound_tts_config(audio_output):
    return {
        "voice_tts": {
            "enabled": True,
            "bundle_path": "models/zipvoice",
            "deployment": "ubuntu_onnx",
            "audio_output": audio_output,
        },
    }


def _playback_parameters(nodes):
    player = next(node for node in nodes if node._Node__node_executable == "audio_playback_node")
    parameters = {}
    for raw_key, raw_value in player._Node__parameters[0].items():
        key = "".join(getattr(item, "text", str(item)) for item in raw_key)
        if isinstance(raw_value, tuple):
            raw_value = "".join(getattr(item, "text", str(item)) for item in raw_value).strip()
            if raw_value.endswith("\n..."):
                raw_value = raw_value[:-4].rstrip()
        parameters[key] = raw_value
    return parameters


def test_voice_tts_reads_topic_and_format_from_the_binding(tmp_path, monkeypatch):
    config = _bound_tts_config(_bound_audio_output(sample_rate_hz=24000, channels=1))

    nodes = voice_tts.generate_voice_tts_nodes(config)

    parameters = _playback_parameters(nodes)
    assert parameters["audio_topic"] == "/audio/play"
    assert parameters["playback_sample_rate"] == 24000
    assert parameters["playback_channels"] == 1


def test_voice_tts_tracks_the_runtime_declared_playback_rate(tmp_path, monkeypatch):
    """A vendor runtime declaring 16 kHz playback overrides the legacy default."""
    config = _bound_tts_config(_bound_audio_output(sample_rate_hz=16000, channels=1))

    nodes = voice_tts.generate_voice_tts_nodes(config)

    parameters = _playback_parameters(nodes)
    assert parameters["playback_sample_rate"] == 16000


def test_voice_tts_rejects_an_unbound_audio_output(tmp_path, monkeypatch):
    config = _bound_tts_config({"interface": "audio.play", "type": "audio_common_msgs/msg/AudioDataStamped"})
    with pytest.raises(ValueError, match="must be bound to a runtime description"):
        voice_tts.generate_voice_tts_nodes(config)


def test_voice_tts_without_a_binding_keeps_the_audio_io_path(tmp_path, monkeypatch):
    config = {
        "audio_io": {"enabled": True, "playback_topic": "/audio/play", "playback_sample_rate": 24000},
        "voice_tts": {"enabled": True, "bundle_path": "models/zipvoice", "deployment": "ubuntu_onnx"},
    }

    nodes = voice_tts.generate_voice_tts_nodes(config)

    parameters = _playback_parameters(nodes)
    assert parameters["audio_topic"] == "/audio/play"
    assert parameters["playback_sample_rate"] == 24000


def test_voice_tts_publishes_audio_data_unless_the_config_names_the_speaker_type(tmp_path, monkeypatch):
    # A bound speaker keeps receiving AudioData unless the deployment names another type: the
    # ALSA speaker is projected as AudioDataStamped while audio_play is documented to take AudioData.
    config = _bound_tts_config(_bound_audio_output())
    default = _playback_parameters(voice_tts.generate_voice_tts_nodes(config))
    assert default["audio_message_type"] == "audio_common_msgs/msg/AudioData"

    config["voice_tts"]["audio_message_type"] = "audio_common_msgs/msg/AudioDataStamped"
    stamped = _playback_parameters(voice_tts.generate_voice_tts_nodes(config))
    assert stamped["audio_message_type"] == "audio_common_msgs/msg/AudioDataStamped"


def test_voice_tts_resampling_to_the_speaker_rate_is_opt_in(tmp_path, monkeypatch):
    config = _bound_tts_config(_bound_audio_output(sample_rate_hz=16000, channels=1))
    assert _playback_parameters(voice_tts.generate_voice_tts_nodes(config))["resample_to_playback_rate"] is False

    config["voice_tts"]["resample_to_playback_rate"] = True
    assert _playback_parameters(voice_tts.generate_voice_tts_nodes(config))["resample_to_playback_rate"] is True


def _decode_launch_json_string(raw_value):
    """Launch quotes a string that holds YAML-special characters; strip that before parsing."""
    normalized = raw_value.strip()
    if len(normalized) >= 2 and normalized[0] == normalized[-1] and normalized[0] in {"'", '"'}:
        normalized = normalized[1:-1]
    import json

    return json.loads(normalized)


def test_voice_tts_forwards_the_bound_speaker_qos(tmp_path, monkeypatch):
    """The speaker's subscription QoS is part of the contract: the X2 runtime subscribes best_effort."""
    speaker_qos = {"reliability": "best_effort", "durability": "volatile", "history": "keep_last", "depth": 10}
    config = _bound_tts_config(_bound_audio_output(qos=speaker_qos))

    parameters = _playback_parameters(voice_tts.generate_voice_tts_nodes(config))

    assert _decode_launch_json_string(parameters["audio_qos_json"]) == speaker_qos


def test_voice_tts_without_a_declared_qos_keeps_the_node_default(tmp_path, monkeypatch):
    # A speaker that declares no qos must keep the node's rclpy default rather than inherit one.
    config = _bound_tts_config(_bound_audio_output())

    assert _playback_parameters(voice_tts.generate_voice_tts_nodes(config))["audio_qos_json"].strip("'") == ""


def test_voice_tts_builder_passes_num_steps_to_the_session(monkeypatch, tmp_path):
    """The flow step count must be reachable from YAML.

    It is ~98% of synthesis time and linear in the step count, but it lives in
    a digest-verified bundle asset, so the only way to pick a different
    latency/quality point is to send it as a runtime option.
    """
    from voice_tts_service.defaults import VOICE_TTS_DEFAULTS

    monkeypatch.setattr(voice_tts, "_load_voice_tts_service", lambda: VOICE_TTS_DEFAULTS)

    nodes = voice_tts.generate_voice_tts_nodes(
        {
            "voice_tts": {
                "enabled": True,
                "bundle_path": str(tmp_path / "bundle"),
                "deployment": "ubuntu_onnx",
                "num_steps": 2,
            },
            "audio_io": {"enabled": True},
        }
    )
    options = _runtime_options(nodes[0])

    assert options["num_steps"] == 2


def test_voice_tts_builder_leaves_num_steps_to_the_bundle_when_unset(monkeypatch, tmp_path):
    """Unset in YAML, the session uses the step count its bundle ships with.

    Fewer steps is a measured trade a profile opts into; it must not change
    every deployment that never asked for it.
    """
    from voice_tts_service.defaults import VOICE_TTS_DEFAULTS

    monkeypatch.setattr(voice_tts, "_load_voice_tts_service", lambda: VOICE_TTS_DEFAULTS)

    nodes = voice_tts.generate_voice_tts_nodes(
        {
            "voice_tts": {
                "enabled": True,
                "bundle_path": str(tmp_path / "bundle"),
                "deployment": "ubuntu_onnx",
            },
            "audio_io": {"enabled": True},
        }
    )
    options = _runtime_options(nodes[0])

    assert "num_steps" not in options
def _additional_env(node):
    """Node.additional_env holds (name, value) substitution pairs, not plain strings."""

    def text(value):
        return "".join(getattr(item, "text", str(item)) for item in value)

    return {text(name): text(value) for name, value in (node.additional_env or ())}


def _manifest_with_profiles(monkeypatch, *, role_runtime_profiles=None, runtime_profile=None):
    import sys
    import types

    deployment = types.SimpleNamespace(role_runtime_profiles=role_runtime_profiles, runtime_profile=runtime_profile)
    module = types.ModuleType("inference_manifest")
    module.load_inference_manifest = lambda *_args, **_kwargs: types.SimpleNamespace(deployment=deployment)
    monkeypatch.setitem(sys.modules, "inference_manifest", module)


def _model_host(nodes):
    return next(node for node in nodes if node._Node__node_executable == "model_service_node")


def test_ascend_deployment_runs_under_the_cann_abi_its_om_declares(monkeypatch, tmp_path):
    """The OM is built against one CANN ABI; the node must not inherit the caller's toolkit."""
    from types import SimpleNamespace

    from voice_tts_service import cann_runtime

    toolkit = tmp_path / "cann" / "8.1.RC1"
    (toolkit / "lib64").mkdir(parents=True)
    (tmp_path / "cann" / "latest").mkdir()
    monkeypatch.setattr(cann_runtime, "CANN_INSTALL_ROOT", tmp_path / "cann")
    ascend = SimpleNamespace(backend="ascend", target=SimpleNamespace(runtime_abi="cann-8.1.RC1"))
    _manifest_with_profiles(monkeypatch, role_runtime_profiles={"text_encoder": ascend})

    host = _model_host(voice_tts.generate_voice_tts_nodes(_bound_tts_config(_bound_audio_output())))

    assert _additional_env(host)["ASCEND_TOOLKIT_HOME"] == str(toolkit)


def test_non_ascend_deployment_gets_no_cann_environment(monkeypatch, tmp_path):
    from types import SimpleNamespace

    _manifest_with_profiles(monkeypatch, runtime_profile=SimpleNamespace(backend="onnxruntime", target=None))

    host = _model_host(voice_tts.generate_voice_tts_nodes(_bound_tts_config(_bound_audio_output())))

    assert _additional_env(host) == {}


def test_the_playback_node_writes_pcm_the_way_the_bound_speaker_asks(tmp_path, monkeypatch):
    paced_output = _bound_audio_output(write_chunk_ms=200, prebuffer_ms=1000)
    paced = _playback_parameters(voice_tts.generate_voice_tts_nodes(_bound_tts_config(paced_output)))
    assert (paced["chunk_ms"], paced["prebuffer_ms"]) == (200, 1000)

    # A speaker that declares nothing leaves the node on its own defaults.
    unpaced = _playback_parameters(voice_tts.generate_voice_tts_nodes(_bound_tts_config(_bound_audio_output())))
    assert "chunk_ms" not in unpaced and "prebuffer_ms" not in unpaced
