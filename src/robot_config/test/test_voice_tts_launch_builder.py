import pytest

from robot_config.launch_builders import voice_tts


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
    # Without a runtime binding the local audio_play contract is unchanged.
    assert playback["audio_message_type"] == "audio_common_msgs/msg/AudioData"
    assert playback["resample_to_playback_rate"] is False
    assert playback["chunk_ms"] == 50


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


def _additional_env(node):
    """Node.additional_env holds (name, value) substitution pairs, not plain strings."""

    def text(value):
        return "".join(getattr(item, "text", str(item)) for item in value)

    return {text(name): text(value) for name, value in (node.additional_env or ())}


_PLAYBACK_DEFAULTS = {
    "enabled": False,
    "bundle_path": "models/zipvoice",
    "deployment": "",
    "exit_on_init_failure": True,
}


def _bound_audio_output(topic="/audio/play"):
    return {
        "interface": "audio.play",
        "type": "audio_common_msgs/msg/AudioDataStamped",
        "topic": topic,
        "_interface_source": {"id": "audio.play"},
    }


def _runtime_playback_config(tmp_path, **voice_tts_overrides):
    config = {
        "enabled": True,
        "bundle_path": str(tmp_path),
        "deployment": "ascend_310p",
        "audio_output": _bound_audio_output(),
    }
    config.update(voice_tts_overrides)
    return {
        "voice_tts": config,
        "audio_io": {"enabled": True, "source": "runtime", "playback_sample_rate": 16000, "playback_channels": 1},
    }


def test_runtime_bound_playback_publishes_on_the_described_endpoint(monkeypatch, tmp_path):
    monkeypatch.setattr(voice_tts, "_load_voice_tts_service", lambda: _PLAYBACK_DEFAULTS)
    config = _runtime_playback_config(
        tmp_path, audio_output=_bound_audio_output("/instance_x/play"), resample_to_playback_rate=True, chunk_ms=100
    )

    playback = _node_parameters(voice_tts.generate_voice_tts_nodes(config)[1])

    assert playback["audio_topic"] == "/instance_x/play"
    assert playback["audio_message_type"] == "audio_common_msgs/msg/AudioDataStamped"
    assert playback["playback_sample_rate"] == 16000
    assert playback["resample_to_playback_rate"] is True
    assert playback["chunk_ms"] == 100


def test_unbound_audio_output_is_rejected(monkeypatch, tmp_path):
    # Declaring the logical sink but launching before the runtime resolved it must
    # fail here rather than publish to a guessed topic.
    monkeypatch.setattr(voice_tts, "_load_voice_tts_service", lambda: _PLAYBACK_DEFAULTS)
    config = _runtime_playback_config(
        tmp_path, audio_output={"interface": "audio.play", "type": "audio_common_msgs/msg/AudioDataStamped"}
    )

    with pytest.raises(ValueError, match="must be bound to a runtime description"):
        voice_tts.generate_voice_tts_nodes(config)


def test_explicit_playback_topic_conflicting_with_the_binding_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(voice_tts, "_load_voice_tts_service", lambda: _PLAYBACK_DEFAULTS)
    config = _runtime_playback_config(tmp_path)
    config["audio_io"]["playback_topic"] = "/audio/other_play"

    with pytest.raises(ValueError, match="conflicts with the bound runtime endpoint"):
        voice_tts.generate_voice_tts_nodes(config)


def test_ascend_deployment_runs_under_the_cann_abi_its_om_declares(monkeypatch, tmp_path):
    """The OM is built against one CANN ABI; the node must not inherit the caller's toolkit."""
    monkeypatch.setattr(voice_tts, "_load_voice_tts_service", lambda: _PLAYBACK_DEFAULTS)
    toolkit = tmp_path / "cann" / "8.1.RC1"
    (toolkit / "lib64").mkdir(parents=True)
    monkeypatch.setattr(voice_tts, "resolve_cann_root", lambda abi: toolkit if abi == "cann-8.1.RC1" else tmp_path)

    class _Target:
        runtime_abi = "cann-8.1.RC1"

    class _Profile:
        backend = "ascend"
        target = _Target()

    class _Deployment:
        role_runtime_profiles = {"text_encoder": _Profile()}
        runtime_profile = None

    import sys
    import types

    module = types.ModuleType("inference_manifest")
    module.load_inference_manifest = lambda *_a, **_k: types.SimpleNamespace(deployment=_Deployment())
    monkeypatch.setitem(sys.modules, "inference_manifest", module)

    host, _player = voice_tts.generate_voice_tts_nodes(_runtime_playback_config(tmp_path))

    assert _additional_env(host)["ASCEND_TOOLKIT_HOME"] == str(toolkit)


def test_non_ascend_deployment_gets_no_cann_environment(monkeypatch, tmp_path):
    monkeypatch.setattr(voice_tts, "_load_voice_tts_service", lambda: _PLAYBACK_DEFAULTS)

    class _Profile:
        backend = "torch"
        target = None

    class _Deployment:
        role_runtime_profiles = None
        runtime_profile = _Profile()

    import sys
    import types

    module = types.ModuleType("inference_manifest")
    module.load_inference_manifest = lambda *_a, **_k: types.SimpleNamespace(deployment=_Deployment())
    monkeypatch.setitem(sys.modules, "inference_manifest", module)

    host, _player = voice_tts.generate_voice_tts_nodes(_runtime_playback_config(tmp_path))

    assert _additional_env(host) == {}
