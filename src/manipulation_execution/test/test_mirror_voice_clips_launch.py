"""The mirror_session builder and ``internvl_voice`` (feat/x2-internvl-voice).

Off (no section, or ``enabled: false``) the voice node's parameters are the
ones of 676fa608, key for key and byte for byte; on, they gain ``clips_json``
and nothing else changes. A broken section still gives a ``clips_json`` that
plays the default clips, and ``internvl_voice`` without the voice node warns
and starts nothing.
"""

from __future__ import annotations

import json
import os
import wave
from pathlib import Path

import pytest
from launch_ros.actions import Node
from test_mirror_session_launch_builder import _config, _node_parameters

import manipulation_execution.launch_builders.mirror_session as builder
from embodied_common.internvl_voice_contract import parse_playback_json
from manipulation_execution.launch_builders.mirror_session import generate_mirror_session_nodes
from manipulation_execution.mirror_voice.speaker import load_clips

REPO = Path(__file__).resolve().parents[3]
ROBOT_CONFIG_SRC = REPO / "src" / "robot_config"

# The voice node's parameters at 676fa608 for ``config_with_voice()`` below, in order.
GOLDEN_676FA608 = [
    (
        "voice_json",
        '{"enabled": true, "phrases": {"goodbye": ["下次再见"], "mirroring": ["开始模仿啦"], '
        '"ready": ["我准备好咯"], "starting": ["启动中"]}, "queue_max": 8, '
        '"utterance_topic": "/visual_interaction/utterance"}',
    ),
    ("tts_service", "/voice_tts/synthesize"),
    ("playback_service", "/voice_tts/play"),
    ("tts_timeout_sec", 15.0),
    ("playback_timeout_sec", 300.0),
    ("status_period_sec", 1.0),
]


def config_with_voice(internvl=None, *, voice_enabled=True):
    config = _config()
    config["voice_tts"] = {"enabled": True, "deployment": "/opt/zipvoice"}
    config["mirror_session"]["voice"] = {
        "enabled": voice_enabled,
        "queue_max": 8,
        "utterance_topic": "/visual_interaction/utterance",
    }
    if internvl is not None:
        config["internvl_voice"] = internvl
    return config


def voice_node(actions):
    nodes = [node for node in actions if isinstance(node, Node) and node._Node__node_executable == "mirror_voice_node"]
    return nodes[0] if nodes else None


def raw_parameters(node):
    """The parameters as handed to launch: substitution text, not decoded."""

    def text(value):
        if not isinstance(value, tuple):
            return value
        return "".join(item.text if hasattr(item, "text") else str(item) for item in value)

    return [(text(key), text(value)) for key, value in node._Node__parameters[0].items()]


@pytest.fixture
def installed(monkeypatch, tmp_path):
    """robot_config's share is its source tree; WORKSPACE a temp dir with the WAVs of every clip."""
    monkeypatch.setattr(builder, "_find_share", lambda package: str(ROBOT_CONFIG_SRC))
    monkeypatch.setenv("WORKSPACE", str(tmp_path))
    return tmp_path


class Log:
    def __init__(self):
        self.records: list[tuple[str, str]] = []

    def __iter__(self):
        return iter(self.records)

    def info(self, message):
        self.records.append(("info", str(message)))

    def warning(self, message):
        self.records.append(("warning", str(message)))

    def error(self, message):
        self.records.append(("error", str(message)))


@pytest.fixture
def log(monkeypatch):
    """The builder's logger, recorded."""
    recorded = Log()
    monkeypatch.setattr(builder, "logger", recorded)
    return recorded


def write_wavs(folder, ids, suffix):
    folder.mkdir(parents=True, exist_ok=True)
    for clip_id in ids:
        with wave.open(str(folder / f"{clip_id}{suffix}"), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(24000)
            stream.writeframes(b"\x00\x00" * 240)


class TestOff:
    @pytest.mark.parametrize("internvl", [None, {"enabled": False}], ids=["absent", "disabled"])
    def test_the_voice_node_parameters_are_those_of_676fa608(self, internvl, installed):
        node = voice_node(generate_mirror_session_nodes(config_with_voice(internvl)))

        assert [(key, value) for key, value in _node_parameters(node).items()] == GOLDEN_676FA608

    def test_off_is_byte_identical_to_absent_for_the_whole_launch(self, installed):
        absent = generate_mirror_session_nodes(config_with_voice())
        off = generate_mirror_session_nodes(config_with_voice({"enabled": False}))

        assert [type(action) for action in absent] == [type(action) for action in off]
        for a, b in zip(absent, off, strict=True):
            if isinstance(a, Node):
                assert raw_parameters(a) == raw_parameters(b)
        assert "clips_json" not in dict(raw_parameters(voice_node(off)))


class TestOn:
    def test_on_adds_clips_json_and_changes_nothing_else(self, installed):
        node = voice_node(generate_mirror_session_nodes(config_with_voice({"enabled": True})))
        parameters = _node_parameters(node)

        assert [(key, value) for key, value in parameters.items() if key != "clips_json"] == GOLDEN_676FA608
        playback, warnings = parse_playback_json(parameters["clips_json"])
        assert warnings == []
        assert playback["clip_dirs"][0] == os.path.join(str(installed), "assets/audio/x2_mirror_clips_v1")
        assert all(playback["events"][event] for event in ("starting", "ready", "mirroring", "goodbye"))
        assert playback["standby"]["ids"]
        assert playback["asr"] == {"mode": "mute", "resume_delay_sec": 0.8, "topic": "/voice_control"}

    def test_the_default_table_is_the_installed_robot_config_one(self, installed, log):
        generate_mirror_session_nodes(config_with_voice({"enabled": True}))

        table = str(ROBOT_CONFIG_SRC / "config" / "mirror_voice" / "x2_clips_v1.yaml")
        assert any("prerecorded clips only" in message and table in message for _, message in log)

    @pytest.mark.parametrize("why", ["off", "absent"])
    def test_without_the_voice_node_it_warns_and_starts_none(self, why, installed, log):
        config = config_with_voice({"enabled": True}, voice_enabled=False)
        if why == "absent":
            del config["mirror_session"]["voice"]
        actions = generate_mirror_session_nodes(config)

        assert voice_node(actions) is None
        assert len(actions) == 2  # the session node and its release handler
        assert any(level == "warning" and message.startswith("internvl_voice is on") for level, message in log)

    def test_voice_off_and_internvl_off_warns_nothing_about_clips(self, installed, log):
        generate_mirror_session_nodes(config_with_voice({"enabled": False}, voice_enabled=False))

        assert not any("internvl_voice" in message for _, message in log)


class TestBrokenConfig:
    def test_bad_items_keep_their_defaults_and_the_clips_still_play(self, installed, log):
        broken = {
            "enabled": "yes",
            "clips_table": "/nowhere/table.yaml",
            "asr": "loud",
            "asr_resume_delay_sec": -3,
            "standby": {"first_after_sec": "soon", "gap_sec": [9, 1]},
            "look": 5,
            "nonsense": True,
        }
        node = voice_node(generate_mirror_session_nodes(config_with_voice(broken)))
        warned = " ".join(message for level, message in log if level == "warning")
        for item in (
            "enabled",
            "clips_table",
            "asr",
            "asr_resume_delay_sec",
            "first_after_sec",
            "gap_sec",
            "look",
            "nonsense",
        ):
            assert item in warned

        playback, _ = parse_playback_json(_node_parameters(node)["clips_json"])
        assert playback["asr"] == {"mode": "mute", "resume_delay_sec": 0.8, "topic": "/voice_control"}
        assert playback["standby"]["first_after_sec"] == (1.0, 3.0)
        assert playback["standby"]["gap_sec"] == (2.0, 5.0)

        write_wavs(Path(playback["clip_dirs"][0]), playback["clips"], playback["wav_suffix"])
        clips, report = load_clips(playback)
        assert report.found == report.total > 0
        assert report.silent_events == ()
        assert all(clips.events[event] for event in ("starting", "ready", "mirroring", "goodbye"))
        assert clips.standby

    def test_a_section_that_is_not_a_mapping_still_plays_the_default_clips(self, installed):
        node = voice_node(generate_mirror_session_nodes(config_with_voice("on please")))

        playback, warnings = parse_playback_json(_node_parameters(node)["clips_json"])
        assert warnings == []
        assert playback["clips"]

    def test_a_build_bundle_crash_falls_back_to_the_defaults(self, installed, monkeypatch):
        import embodied_common.internvl_voice_contract as contract

        real = contract.build_bundle

        def fragile(raw, *args):
            if raw != {"enabled": True}:
                raise RuntimeError("boom")
            return real(raw, *args)

        monkeypatch.setattr(contract, "build_bundle", fragile)
        node = voice_node(generate_mirror_session_nodes(config_with_voice({"enabled": True, "asr": "off"})))

        playback, _ = parse_playback_json(_node_parameters(node)["clips_json"])
        assert playback["asr"]["mode"] == "mute"
        assert playback["clips"]

    def test_no_robot_config_share_still_launches(self, monkeypatch, tmp_path):
        def missing(package):
            raise LookupError(package)

        monkeypatch.setattr(builder, "_find_share", missing)
        monkeypatch.setenv("WORKSPACE", str(tmp_path))
        actions = generate_mirror_session_nodes(config_with_voice({"enabled": True}))

        assert "clips_json" in _node_parameters(voice_node(actions))
        assert json.loads(_node_parameters(voice_node(actions))["clips_json"])["clips"] == {}
