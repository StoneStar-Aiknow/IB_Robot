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
import pytest

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
            # declared by generate_launch_description; a direct _launch_setup
            # caller must supply it (runtime-deployment-peripherals D7)
            "peripherals": "",
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


def test_a_non_empty_deployment_peripherals_fragment_is_rejected(tmp_path):
    """runtime-deployment-peripherals D7: the X2's cameras are vendor public
    interfaces; a non-empty fragment fails closed in launch setup with the
    contract message instead of being silently ignored."""
    fragment = tmp_path / "deployment_peripherals.yaml"
    fragment.write_text("peripherals:\n- type: camera\n  name: top\n  width: 1280\n", encoding="utf-8")
    context = launch.LaunchContext()
    context.launch_configurations.update(
        {
            "profile": str(_PROFILE),
            "simulated": "false",
            "use_sim": "",
            "sim_platform": "",
            "instance_id": "",
            "initial_mode": "",
            "peripherals": str(fragment),
        }
    )
    with pytest.raises(
        RuntimeError, match="aimdk_robot exposes vendor interfaces and accepts no deployment peripherals"
    ):
        _runtime_launch_module()._launch_setup(context)


def test_launch_declares_the_peripherals_argument():
    """D7 (m3): the provider's real launch description declares the
    peripherals argument every provider contract requires (AimDK rejects a
    non-empty value, but the shared argument must exist)."""
    from launch.actions import DeclareLaunchArgument

    def text(value):
        return "".join(part.text if hasattr(part, "text") else str(part) for part in value)

    description = _runtime_launch_module().generate_launch_description()
    names = {
        text(action._DeclareLaunchArgument__name)
        for action in description.entities
        if isinstance(action, DeclareLaunchArgument)
    }
    assert "peripherals" in names
