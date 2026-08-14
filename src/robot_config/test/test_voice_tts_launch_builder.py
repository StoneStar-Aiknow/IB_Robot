from robot_config.launch_builders import voice_tts


def test_voice_tts_builder_does_not_open_bundle_before_node_start(monkeypatch, tmp_path):
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
                "bundle_path": str(tmp_path / "not-mounted-yet"),
                "deployment": "ascend_310p",
                "exit_on_init_failure": False,
            }
        }
    )

    assert len(nodes) == 2
