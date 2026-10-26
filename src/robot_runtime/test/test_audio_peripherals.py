"""Audio peripheral ownership tests (openspec change ``runtime-owned-audio-io``).

Microphones and speakers are host devices owned by the robot runtime: the
peripheral layer generates the shared audio_common ALSA nodes (the same
generation source robot_config's legacy provider-less ``audio_io`` entry
calls), projects ``audio.capture`` / ``audio.play`` interfaces with format
metadata, and derives the ``interaction.audio_*`` capability entries.
"""

from __future__ import annotations

import launch_ros.actions
import yaml

from robot_runtime.interface_description import (
    InterfaceDescriptionError,
    build_description,
    peripheral_interfaces,
)
from robot_runtime.launch_support import peripheral_stack_actions
from robot_runtime.peripherals import (
    audio_capabilities,
    audio_capture_node,
    audio_nodes,
    audio_play_node,
    peripheral_nodes,
)

_RESPEAKER = {
    "type": "microphone",
    "name": "respeaker",
    "driver": "alsa",
    "params": {
        "device": "hw:0,0",
        "channels": 6,
        "sample_rate": 16000,
        "sample_format": "S16LE",
        "channel_indices": [1, 2, 3, 4],
    },
}

_SPEAKER = {
    "type": "speaker",
    "name": "speaker",
    "driver": "alsa",
    "params": {"device": "", "channels": 1, "sample_rate": 24000, "sample_format": "S16LE"},
}


def _text(value):
    return "".join(item.text if hasattr(item, "text") else str(item) for item in value)


def _decode(value):
    return yaml.safe_load(_text(value)) if isinstance(value, tuple | list) else value


def _params(node):
    return {_text(key): _decode(value) for key, value in node._Node__parameters[0].items()}


def _remappings(node):
    return [(_text(source), _text(target)) for source, target in node._Node__remappings]


def _node_by_name(nodes, name):
    matches = [node for node in nodes if getattr(node, "_Node__node_name", None) == name]
    assert len(matches) == 1, f"expected exactly one node named {name!r}"
    return matches[0]


def test_audio_capture_node_parameters_and_topics():
    node = audio_capture_node(_RESPEAKER["params"])
    assert isinstance(node, launch_ros.actions.Node)
    assert (node.node_package, node.node_executable, node._Node__node_name) == (
        "audio_capture",
        "audio_capture_node",
        "audio_capture",
    )
    assert _params(node) == {
        "src": "alsasrc",
        "dst": "appsink",
        "format": "wave",
        "device": "hw:0,0",
        "channels": 6,
        "sample_rate": 16000,
        "sample_format": "S16LE",
        "depth": 16,
    }
    assert _remappings(node) == [
        ("audio", "/audio/capture"),
        ("audio_stamped", "/audio/capture_stamped"),
        ("audio_info", "/audio/info"),
    ]


def test_audio_capture_node_defaults_match_the_legacy_audio_io_entry():
    node = audio_capture_node({})
    assert _params(node) == {
        "src": "alsasrc",
        "dst": "appsink",
        "format": "wave",
        "device": "default",
        "channels": 6,
        "sample_rate": 16000,
        "sample_format": "S16LE",
        "depth": 16,
    }


def test_audio_play_node_parameters_and_topic():
    node = audio_play_node(_SPEAKER["params"])
    assert (node.node_package, node.node_executable, node._Node__node_name) == (
        "audio_play",
        "audio_play_node",
        "audio_play",
    )
    assert _params(node) == {
        "dst": "alsasink",
        "device": "",
        "format": "wave",
        "channels": 1,
        "depth": 16,
        "sample_rate": 24000,
        "sample_format": "S16LE",
    }
    assert _remappings(node) == [("audio", "/audio/play")]


def test_audio_play_node_defaults_match_the_legacy_audio_io_entry():
    node = audio_play_node({})
    assert _params(node)["device"] == ""
    assert _params(node)["channels"] == 1
    assert _params(node)["sample_rate"] == 24000
    assert _params(node)["sample_format"] == "S16LE"


def test_peripheral_nodes_compose_audio_devices_without_tf():
    nodes = peripheral_nodes([_RESPEAKER, _SPEAKER], use_sim=False)
    capture = _node_by_name(nodes, "audio_capture")
    playback = _node_by_name(nodes, "audio_play")
    assert (capture.node_package, capture.node_executable) == ("audio_capture", "audio_capture_node")
    assert (playback.node_package, playback.node_executable) == ("audio_play", "audio_play_node")
    assert all(getattr(node, "node_package", "") != "tf2_ros" for node in nodes), "audio devices carry no frame"


def test_peripheral_stack_actions_runs_audio_in_both_transport_modes():
    profile = {
        "runtime": {"name": "t_robot", "version": "0.1.0"},
        "modes": {"initial": "idle", "idle": {"controllers": []}},
        "capabilities": {},
        "joints": ["1"],
        "controller_manager": "controller_manager",
        "peripherals": [_RESPEAKER, _SPEAKER],
    }
    for simulated in (False, True):
        actions = peripheral_stack_actions(profile, "", simulated)
        names = {getattr(action, "_Node__node_name", "") for action in actions}
        assert "audio_capture" in names, f"simulated={simulated}: the ALSA capture owner must run"
        assert "audio_play" in names, f"simulated={simulated}: the ALSA playback owner must run"


def test_peripheral_interfaces_project_audio_with_format_metadata():
    interfaces = peripheral_interfaces([_RESPEAKER, _SPEAKER])
    capture = interfaces["audio.capture"]
    assert capture["capability"] == "interaction.audio_capture"
    assert capture["direction"] == "publish"
    assert capture["endpoint"] == "/audio/capture_stamped"
    assert capture["message_type"] == "audio_common_msgs/msg/AudioDataStamped"
    assert capture["sample_rate_hz"] == 16000
    assert capture["channels"] == 6
    assert capture["sample_format"] == "S16LE"
    assert capture["channel_indices"] == [1, 2, 3, 4]
    playback = interfaces["audio.play"]
    assert playback["capability"] == "interaction.audio_playback"
    assert playback["direction"] == "subscribe"
    assert playback["endpoint"] == "/audio/play"
    assert playback["sample_rate_hz"] == 24000
    assert playback["channels"] == 1
    assert playback["sample_format"] == "S16LE"
    assert "channel_indices" not in playback


def test_peripheral_interfaces_audio_defaults_without_params():
    interfaces = peripheral_interfaces(
        [
            {"type": "microphone", "name": "mic", "driver": "alsa", "params": {}},
            {"type": "speaker", "name": "spk", "driver": "alsa", "params": {}},
        ]
    )
    assert interfaces["audio.capture"]["sample_rate_hz"] == 16000
    assert interfaces["audio.capture"]["channels"] == 6
    assert interfaces["audio.capture"]["sample_format"] == "S16LE"
    assert "channel_indices" not in interfaces["audio.capture"]
    assert interfaces["audio.play"]["sample_rate_hz"] == 24000
    assert interfaces["audio.play"]["channels"] == 1


def test_audio_interface_endpoint_matches_the_generated_node_topic():
    interfaces = peripheral_interfaces([_RESPEAKER, _SPEAKER])
    nodes = audio_nodes([_RESPEAKER, _SPEAKER])
    capture = _node_by_name(nodes, "audio_capture")
    playback = _node_by_name(nodes, "audio_play")
    assert dict(_remappings(capture))["audio_stamped"] == interfaces["audio.capture"]["endpoint"]
    assert dict(_remappings(playback))["audio"] == interfaces["audio.play"]["endpoint"]


def test_second_audio_device_of_a_kind_is_rejected():
    for periph in (
        {"type": "microphone", "name": "other_mic", "driver": "alsa", "params": {}},
        {"type": "speaker", "name": "other_spk", "driver": "alsa", "params": {}},
    ):
        try:
            peripheral_interfaces([_RESPEAKER, _SPEAKER, periph])
        except InterfaceDescriptionError as exc:
            assert "at most one" in str(exc)
        else:
            raise AssertionError("a second device of the same audio kind must be rejected")


def test_non_alsa_audio_driver_is_rejected():
    for periph in ({"type": "microphone", "name": "m", "driver": "pulse"}, {"type": "speaker", "name": "s"}):
        try:
            peripheral_interfaces([periph])
        except InterfaceDescriptionError as exc:
            assert "unsupported driver" in str(exc)
        else:
            raise AssertionError("non-alsa audio drivers must be rejected")


def _profile(peripherals, capabilities=None):
    return {
        "runtime": {"name": "t_robot", "version": "0.1.0"},
        "modes": {"initial": "idle", "idle": {"controllers": []}},
        "capabilities": capabilities or {},
        "joints": ["1"],
        "peripherals": peripherals,
    }


def test_audio_capabilities_follow_the_capability_schema():
    caps = audio_capabilities([_RESPEAKER, _SPEAKER])
    assert caps == {
        "interaction.audio_capture": {"topics": ["/audio/capture_stamped"], "sample_rate_hz": 16000, "channels": 6},
        "interaction.audio_playback": {"endpoints": ["/audio/play"], "sample_rate_hz": 24000, "channels": 1},
    }


def test_build_description_accepts_matching_declared_audio_capability():
    capabilities = {
        "interaction.audio_capture": {"topics": ["/audio/capture_stamped"], "sample_rate_hz": 16000, "channels": 6}
    }
    description = build_description(_profile([_RESPEAKER], capabilities), use_sim=True)
    assert description["interfaces"]["audio.capture"]["channels"] == 6


def test_build_description_rejects_contradicting_declared_audio_capability():
    capabilities = {
        "interaction.audio_playback": {"endpoints": ["/audio/play"], "sample_rate_hz": 16000, "channels": 1}
    }
    try:
        build_description(_profile([_SPEAKER], capabilities), use_sim=True)
    except InterfaceDescriptionError as exc:
        assert "contradicts the audio peripherals" in str(exc)
    else:
        raise AssertionError("a profile capability contradicting the audio peripherals must be rejected")
