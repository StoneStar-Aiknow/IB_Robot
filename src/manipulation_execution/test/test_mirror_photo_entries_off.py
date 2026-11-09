"""Integration decision: mirror photos never start under the grasp or summit entries.

feat/x2-photo-capture turns ``mirror_session.photo`` on in
aimdk_x2_voice_mirror_all.yaml. aimdk_x2_voice_mirror_grasp.yaml (built on it)
and aimdk_x2_summit.yaml (built on grasp) say ``photo: enabled: false`` so
those deployments launch no ``mirror_photo_node``. voice_mirror_all keeps it on
(the positive control below).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from launch_ros.actions import Node
from test_mirror_session_launch_builder import _config

from manipulation_execution.launch_builders.mirror_photo import photo_parameters
from manipulation_execution.launch_builders.mirror_session import generate_mirror_session_nodes
from robot_config.loader import _load_robot_section

REPO = Path(__file__).resolve().parents[3]
ROBOTS = REPO / "src" / "robot_config" / "config" / "robots"
STREAM = {
    "enabled": True,
    "frame_topic": "/stream/pear",
    "status_topic": "/stream/status",
    "actor_state_topic": "/stream/actor",
    "frame_source": {"interface": "camera.head.color_compressed", "decoder": "mock_cpu"},
    "yolox_detect_service": "/perception/yolox",
    "pear_parameters_service": "/perception/pear",
}


def _merged(name: str) -> dict:
    return _load_robot_section(ROBOTS / name)[1]


def _photo_nodes(photo_section) -> list:
    config = _config(photo=photo_section)
    config["motion_imitation_stream"] = dict(STREAM)
    actions = generate_mirror_session_nodes(config)
    return [a for a in actions if isinstance(a, Node) and a._Node__node_executable == "mirror_photo_node"]


@pytest.mark.parametrize("name", ["aimdk_x2_voice_mirror_grasp.yaml", "aimdk_x2_summit.yaml"])
def test_grasp_and_summit_start_no_photo_node(name):
    config = _merged(name)
    section = config["mirror_session"]["photo"]
    assert section["enabled"] is False
    assert photo_parameters(config, config["mirror_session"], {"camera_topic": "/c"}, lambda _text: None) is None
    assert _photo_nodes(section) == []


def test_voice_mirror_all_still_starts_one():
    section = _merged("aimdk_x2_voice_mirror_all.yaml")["mirror_session"]["photo"]
    assert section["enabled"] is True
    assert len(_photo_nodes(section)) == 1
