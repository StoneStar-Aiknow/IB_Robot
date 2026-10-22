"""robot.launch.py step 9: a voice TTS that cannot be launched stops the launch only when it is meant to."""

import importlib.util
from pathlib import Path

import pytest

from robot_config.launch_builders import voice_tts


def _robot_launch():
    path = Path(__file__).parents[1] / "launch" / "robot.launch.py"
    spec = importlib.util.spec_from_file_location("voice_tts_robot_launch", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _broken(_config):
    raise ValueError("WORKSPACE must be an absolute path")


def test_voice_tts_actions_pass_through(monkeypatch):
    marker = object()
    robot_launch = _robot_launch()
    monkeypatch.setattr(voice_tts, "generate_voice_tts_nodes", lambda _config: [marker])

    assert robot_launch._generate_voice_tts_actions({"voice_tts": {"enabled": True}}) == [marker]


def test_voice_tts_failure_is_skipped_when_exit_on_init_failure_is_false(monkeypatch):
    robot_launch = _robot_launch()
    monkeypatch.setattr(voice_tts, "generate_voice_tts_nodes", _broken)

    config = {"voice_tts": {"enabled": True, "exit_on_init_failure": False}}

    assert robot_launch._generate_voice_tts_actions(config) == []


@pytest.mark.parametrize(
    "config",
    [
        {"voice_tts": {"enabled": True, "exit_on_init_failure": True}},
        {"voice_tts": {"enabled": True}},
        {"voice_tts": {"enabled": True, "exit_on_init_failure": "false"}},
        {"voice_tts": None},
        {},
    ],
    ids=["true", "default", "string", "none", "missing"],
)
def test_voice_tts_failure_stops_the_launch_otherwise(monkeypatch, config):
    robot_launch = _robot_launch()
    monkeypatch.setattr(voice_tts, "generate_voice_tts_nodes", _broken)

    with pytest.raises(ValueError, match="WORKSPACE"):
        robot_launch._generate_voice_tts_actions(config)
