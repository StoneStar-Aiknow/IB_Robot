"""The motion-template segments pass the data plane's whitelist under the shipped CBF.

The segments are judged by the data plane itself: the voice_mirror_all profile
with template_play on goes through the robot_config builder and this package's
launch, and the node they configure reports how many segments it loaded and
which it left out. A segment refused here would only be found on the board.

fixtures/template_segments_motion_template.json is a copy of the segments file
of feat/x2-motion-template at 0c26dd2b (robot_config/config/motion_template/
segments.json, sha256 143d1303b01f2c98c78fe7e9ad7d7f5a87591ec517c286317e412764d9bdd229,
15 segments). Replace it when that file changes.
"""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
import threading
from pathlib import Path

import launch
from ament_index_python.packages import get_package_prefix
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch_ros.utilities import evaluate_parameters

from test_runtime_audio_launch import _LAUNCH, _nodes, _runtime_launch_module

SEGMENTS = Path(__file__).resolve().parent / "fixtures" / "template_segments_motion_template.json"
ROBOTS = Path(__file__).resolve().parents[4] / "robot_config" / "config" / "robots"
EXPECTED = [
    f"{name}{speed}"
    for name in ("heart", "holy", "overhead_heart", "xsign_left_front", "xsign_right_front")
    for speed in ("", "@50", "@75")
]


def _text(value):
    if isinstance(value, list | tuple):
        return "".join(_text(item) for item in value)
    return str(getattr(value, "text", value))


def _data_plane_parameters(monkeypatch, segments: Path) -> dict:
    """What the voice_mirror_all launch gives the data plane with template_play on."""
    import robot_config.launch_builders.runtime as runtime_module
    from robot_config.loader import _load_robot_section

    _, config = _load_robot_section(ROBOTS / "aimdk_x2_voice_mirror_all.yaml")
    assert config["cbf_safety"]["enabled"] is True
    config["template_play"] = {"enabled": True, "segments_file": str(segments)}
    monkeypatch.setattr(runtime_module, "resolve_runtime_launch", lambda _provider: str(_LAUNCH))
    actions, _waiter = runtime_module.generate_runtime_provider_actions(config, use_sim=False)
    include = next(action for action in actions if isinstance(action, IncludeLaunchDescription))
    arguments = {_text(key): _text(value) for key, value in include.launch_arguments}

    module = _runtime_launch_module()
    context = launch.LaunchContext()
    for action in module.generate_launch_description().entities:
        if isinstance(action, DeclareLaunchArgument) and action.default_value is not None:
            context.launch_configurations[action.name] = launch.utilities.perform_substitutions(
                context, action.default_value
            )
    context.launch_configurations.update(arguments)
    rendered = {
        node._Node__node_name: evaluate_parameters(context, node._Node__parameters)
        for node in _nodes(module._launch_setup(context))
    }
    parameters = {}
    for group in rendered["aimdk_dataplane"]:
        parameters.update(group)
    assert json.loads(parameters["template_play_json"]) == {"enabled": True, "segments_file": str(segments)}
    assert json.loads(parameters["cbf_safety_json"])["mode"] == "enforce"
    return parameters


def _judge(monkeypatch, segments: Path) -> tuple[list[int], list[str]]:
    """The node's own report: segments loaded, and every segment it left out."""
    parameters = _data_plane_parameters(monkeypatch, segments)
    executable = Path(get_package_prefix("aimdk_dataplane")) / "lib" / "aimdk_dataplane" / "aimdk_dataplane"
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", prefix="aimdk_dataplane_", delete=False) as file:
        json.dump({"aimdk_dataplane": {"ros__parameters": parameters}}, file)
    process = subprocess.Popen(
        [str(executable), "--ros-args", "--params-file", file.name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    lines: list[str] = []
    done = threading.Event()

    def read():
        for line in process.stderr:
            lines.append(line.rstrip())
            # The setup's last line: enabled, or it stayed off.
            if "template playback enabled" in line or "template playback stays off" in line:
                done.set()

    threading.Thread(target=read, daemon=True).start()
    try:
        assert done.wait(timeout=20.0), "\n".join(lines[-20:])
    finally:
        process.terminate()
        process.wait(timeout=10.0)
    loaded = [re.search(r"template playback enabled: (\d+) segment", line) for line in lines]
    refused = [line for line in lines if "template segment left out" in line or "template whitelist" in line]
    return [int(match.group(1)) for match in loaded if match], refused


def test_every_motion_template_segment_passes_the_whitelist_under_voice_mirror_all(monkeypatch):
    assert sorted(json.loads(SEGMENTS.read_text(encoding="utf-8"))["segments"]) == sorted(EXPECTED)
    loaded, refused = _judge(monkeypatch, SEGMENTS)
    assert refused == []
    assert loaded == [len(EXPECTED)]


def test_the_same_check_refuses_holy_once_its_target_moves_further(monkeypatch, tmp_path):
    """Control: the check above is live. Holy with its widest joint 0.2 rad further is left out."""
    segments = json.loads(SEGMENTS.read_text(encoding="utf-8"))
    holy = segments["segments"]["holy"]
    joint = max(range(14), key=lambda index: abs(holy["T"][index] - holy["P"][index]))
    holy["T"][joint] += 0.2 if holy["T"][joint] >= holy["P"][joint] else -0.2
    changed = tmp_path / "segments.json"
    changed.write_text(json.dumps(segments), encoding="utf-8")
    loaded, refused = _judge(monkeypatch, changed)
    assert loaded == [len(EXPECTED) - 1]
    assert len(refused) == 1 and "holy" in refused[0] and "holy@" not in refused[0], refused
