"""AimDK X2 runtime launch stays ALSA-free (openspec change ``runtime-owned-audio-io``).

The vendor runtime bridges its own audio; it declares no audio peripherals, so
rendering ``runtime.launch.py`` must never produce ``audio_capture`` /
``audio_play`` nodes, in physical or simulated execution, and the profile's
``audio.capture`` / ``audio.play`` interfaces must carry the format metadata
consumers bind to (mirroring the capability entries).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import launch

_PROFILE = Path(__file__).resolve().parents[1] / "profiles" / "x2_ultra.yaml"
_LAUNCH = Path(__file__).resolve().parents[1] / "launch" / "runtime.launch.py"


def _runtime_launch_module():
    spec = importlib.util.spec_from_file_location("aimdk_robot_runtime_launch_under_test", _LAUNCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _render(simulated: str):
    context = launch.LaunchContext()
    context.launch_configurations.update(
        {
            "profile": str(_PROFILE),
            "simulated": simulated,
            "use_sim": "",
            "sim_platform": "",
            "instance_id": "",
            "initial_mode": "",
        }
    )
    return _runtime_launch_module()._launch_setup(context)


def _flatten(actions, out=None):
    out = out if out is not None else []
    for action in actions:
        out.append(action)
        deferred = getattr(action, "_TimerAction__actions", None)
        if deferred:
            _flatten(list(deferred), out)
    return out


def _nodes(actions):
    return [action for action in _flatten(actions) if type(action).__name__ == "Node"]


def test_physical_execution_renders_the_bridge_only():
    actions = _render("false")
    assert [type(action).__name__ for action in actions] == ["Node"]
    nodes = _nodes(actions)
    assert [(node.node_package, node.node_executable, node._Node__node_name) for node in nodes] == [
        ("aimdk_robot", "aimdk_runtime", "aimdk_runtime")
    ]


def test_simulated_execution_declares_no_alsa_audio_nodes():
    actions = _render("true")
    assert [type(action).__name__ for action in actions] == ["Node", "TimerAction", "TimerAction"]
    packages = {node.node_package for node in _nodes(actions)}
    assert not packages & {"audio_capture", "audio_play"}, "the vendor runtime owns its audio"


def test_profile_audio_interfaces_carry_format_metadata():
    from robot_runtime.interface_description import build_description
    from robot_runtime.profile import load_profile

    description = build_description(load_profile(_PROFILE), use_sim=True)
    capture = description["interfaces"]["audio.capture"]
    assert capture["endpoint"] == "/audio/capture_stamped"
    assert capture["sample_rate_hz"] == 16000.0
    assert capture["channels"] == 6
    playback = description["interfaces"]["audio.play"]
    assert playback["endpoint"] == "/audio/play"
    assert playback["sample_rate_hz"] == 16000.0
    assert playback["channels"] == 1
    assert "sample_format" not in capture and "sample_format" not in playback, (
        "the X2 capability entries declare no sample format"
    )
    # The simulated embed stays valid JSON metadata for the runtime node.
    assert json.dumps(description)
