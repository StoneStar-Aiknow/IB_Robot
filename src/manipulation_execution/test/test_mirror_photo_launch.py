"""The mirror_session builder and the photo node (feat/x2-photo-capture).

Without a ``photo`` block, or with ``enabled: false``, the actions are what
they were before the feature: no photo node and the session's parameters
untouched. On, one ``mirror_photo_node`` comes last, with the parameters the
node declares. A section the node would refuse, or no camera, is a WARNING
and the mirror launches without photos. The three X2 entries that carry the
feature (voice_mirror_all, and grasp and summit built on it) have it on, any
entry built on them can turn it off, and the marked block adds nothing but
``mirror_session.photo``.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml
from launch_ros.actions import Node
from test_mirror_session_launch_builder import _config, _node_parameters

import manipulation_execution.launch_builders.mirror_session as builder
from embodied_common.mirror_photo_contract import NODE_NAME
from manipulation_execution.launch_builders.mirror_photo import PHOTO_OFF, photo_parameters
from manipulation_execution.launch_builders.mirror_session import generate_mirror_session_nodes
from manipulation_execution.launch_builders.motion_imitation_stream import mock_hand_frame_topic
from manipulation_execution.mirror_photo.config import from_json, parse_photo_section
from manipulation_execution.mirror_photo_node import PHOTO_PARAMETERS, PhotoSettings
from robot_config.loader import _load_robot_section

REPO = Path(__file__).resolve().parents[3]
ROBOTS = REPO / "src" / "robot_config" / "config" / "robots"
ENTRY = ROBOTS / "aimdk_x2_voice_mirror_all.yaml"
ENTRIES = ("aimdk_x2_voice_mirror_all.yaml", "aimdk_x2_voice_mirror_grasp.yaml", "aimdk_x2_summit.yaml")
BLOCK_START = "    # feat/x2-photo-capture\n"
BLOCK_END = "    # /feat/x2-photo-capture\n"

PHOTO = {
    "enabled": True,
    "photos_dir": "/root/run/photos",
    "target_file": "/root/run/photo_target",
}
STREAM = {
    "enabled": True,
    "frame_topic": "/stream/pear",
    "status_topic": "/stream/status",
    "actor_state_topic": "/stream/actor",
    "frame_source": {"interface": "camera.head.color_compressed", "decoder": "mock_cpu"},
    "yolox_detect_service": "/perception/yolox",
    "pear_parameters_service": "/perception/pear",
}


def config_with(photo=None, *, stream=True):
    config = _config() if photo is None else _config(photo=photo)
    if stream:
        config["motion_imitation_stream"] = dict(STREAM)
    return config


def photo_nodes(actions) -> list:
    return [node for node in actions if isinstance(node, Node) and node._Node__node_executable == "mirror_photo_node"]


def session_parameters(actions) -> dict:
    return _node_parameters(actions[0])


class _Recorder:
    def __init__(self):
        self.lines: list[tuple[str, str]] = []

    def __getattr__(self, level):
        return lambda text: self.lines.append((level, text))


@pytest.fixture
def warnings(monkeypatch):
    """The builder's WARNING lines (its logger is launch's, not one caplog sees)."""
    recorder = _Recorder()
    monkeypatch.setattr(builder, "logger", recorder)
    return lambda: [text for level, text in recorder.lines if level == "warning"]


class TestOff:
    def test_without_the_block_the_actions_are_unchanged(self):
        actions = generate_mirror_session_nodes(config_with())
        assert len(actions) == 2  # the session node and its release handler
        assert photo_nodes(actions) == []

    def test_enabled_false_is_the_same_as_no_block(self):
        off = generate_mirror_session_nodes(config_with({**PHOTO, "enabled": False}))
        none = generate_mirror_session_nodes(config_with())
        assert photo_nodes(off) == []
        assert len(off) == len(none)
        assert session_parameters(off) == session_parameters(none)

    def test_the_session_node_does_not_see_the_photo_node(self):
        on = generate_mirror_session_nodes(config_with(PHOTO))
        none = generate_mirror_session_nodes(config_with())
        assert session_parameters(on) == session_parameters(none)


class TestOn:
    def test_one_photo_node_comes_last(self):
        actions = generate_mirror_session_nodes(config_with(PHOTO))
        (node,) = photo_nodes(actions)
        assert actions[-1] is node
        assert node._Node__node_name == NODE_NAME

    def test_it_reads_the_session_s_camera_and_pear_frames(self):
        (node,) = photo_nodes(generate_mirror_session_nodes(config_with(PHOTO)))
        parameters = _node_parameters(node)
        assert parameters["camera_topic"] == "/camera/head/image_compressed"
        assert parameters["pear_frame_topic"] == "/stream/pear"
        assert parameters["hand_frame_topic"] == ""  # no retarget player following hands here
        assert parameters["status_period_sec"] == 1.0
        section = from_json(json.dumps(parameters["photo_json"]))
        assert section == parse_photo_section(PHOTO)

    def test_the_parameters_are_the_ones_the_node_declares(self):
        (node,) = photo_nodes(generate_mirror_session_nodes(config_with(PHOTO)))
        parameters = photo_parameters(
            config_with(PHOTO),
            config_with(PHOTO)["mirror_session"],
            {"camera_topic": "/c", "pear_frame_topic": "/p", "status_period_sec": 1.0},
            lambda _text: None,
        )
        assert set(_node_parameters(node)) == set(PHOTO_PARAMETERS)
        for name, value in parameters.items():
            assert type(value) is type(PHOTO_PARAMETERS[name]), name  # rclpy refuses another type
        assert PhotoSettings.from_parameters(parameters).section == parse_photo_section(PHOTO)


class TestNeverFailsTheLaunch:
    def test_a_section_the_node_would_refuse_is_a_warning(self, warnings):
        actions = generate_mirror_session_nodes(config_with({**PHOTO, "photos_dir": "relative/photos"}))
        assert photo_nodes(actions) == []
        assert len(actions) == 2
        assert any("photos_dir" in text and PHOTO_OFF in text for text in warnings())

    def test_an_unknown_key_is_a_warning(self, warnings):
        actions = generate_mirror_session_nodes(config_with({**PHOTO, "shutter": True}))
        assert photo_nodes(actions) == []
        assert any("shutter" in text for text in warnings())

    def test_no_camera_is_a_warning(self, warnings):
        actions = generate_mirror_session_nodes(config_with(PHOTO, stream=False))
        assert photo_nodes(actions) == []
        assert any("no compressed camera" in text for text in warnings())

    def test_a_bug_while_working_it_out_is_a_warning(self, monkeypatch, warnings):
        def broken(*_args):
            raise RuntimeError("boom")

        monkeypatch.setattr(builder, "photo_parameters", broken)
        actions = generate_mirror_session_nodes(config_with(PHOTO))
        assert photo_nodes(actions) == []
        assert any("RuntimeError: boom" in text and PHOTO_OFF in text for text in warnings())


# -- the entries ---------------------------------------------------------------------------


def _merged(entry: Path) -> dict:
    return _load_robot_section(entry)[1]


@pytest.mark.parametrize("name", ENTRIES)
def test_the_x2_entries_have_photos_on(name):
    config = _merged(ROBOTS / name)
    section = parse_photo_section(config["mirror_session"]["photo"])
    assert section.enabled
    assert section.photos_dir == "/root/x2-feat-photo-capture-run/photos"
    assert mock_hand_frame_topic(config) == "/hri/hand/frame"


def test_an_entry_built_on_them_turns_photos_off_with_one_line(tmp_path):
    robots = tmp_path / "robots"
    shutil.copytree(ROBOTS, robots)
    entry = robots / "photo_off_probe.yaml"
    entry.write_text(
        yaml.safe_dump(
            {
                "robot": {
                    "base_config": "aimdk_x2_summit.yaml",
                    "name": "photo_off_probe",
                    "mirror_session": {"photo": {"enabled": False}},
                }
            }
        ),
        encoding="utf-8",
    )
    config = _merged(entry)
    assert config["mirror_session"]["photo"]["enabled"] is False
    assert config["mirror_session"]["voice"] == _merged(ROBOTS / "aimdk_x2_summit.yaml")["mirror_session"]["voice"]
    assert photo_parameters(config, config["mirror_session"], {"camera_topic": "/c"}, lambda _text: None) is None


def test_the_block_adds_nothing_but_mirror_session_photo(tmp_path):
    text = ENTRY.read_text(encoding="utf-8")
    assert text.count(BLOCK_START) == 1 and text.count(BLOCK_END) == 1
    start = text.index(BLOCK_START)
    end = text.index(BLOCK_END, start) + len(BLOCK_END)
    robots = tmp_path / "robots"
    shutil.copytree(ROBOTS, robots)
    (robots / ENTRY.name).write_text(text[:start] + text[end:], encoding="utf-8")
    without = _merged(robots / ENTRY.name)
    assert "photo" not in without["mirror_session"]
    current = _merged(ENTRY)
    assert current["mirror_session"].pop("photo")["enabled"] is True
    assert current == without
