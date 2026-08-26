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
