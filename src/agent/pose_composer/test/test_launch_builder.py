"""The pose composer joins a launch only when the robot config enables it, with no literal key."""

import pytest

from pose_composer.launch_builder import generate_pose_composer_nodes

LLM = {"provider": "openai_compatible", "base_url": "http://127.0.0.1:1/v1", "model": "m"}


def _config(tmp_path, **pose):
    mjcf = tmp_path / "scene.xml"
    mjcf.write_text("<mujoco/>")
    block = {"enabled": True, "mjcf_path": str(mjcf), "llm": dict(LLM)}
    block.update(pose)
    return {
        "runtime": {"provider": "aimdk_robot", "profile": "x2_ultra_omnihand"},
        "skill_required_control_mode": "named_motion",
        "control_modes": {"named_motion": {"runtime_mode": "stream"}},
        "pose_composer": block,
    }


def test_disabled_or_absent_launches_nothing(tmp_path):
    assert generate_pose_composer_nodes({}) == []
    assert generate_pose_composer_nodes(_config(tmp_path, enabled=False)) == []


def test_enabled_launches_the_node(tmp_path):
    (node,) = generate_pose_composer_nodes(_config(tmp_path))
    assert node.node_executable == "pose_composer_node"


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"surprise": 1}, "unknown keys"),
        ({"llm": dict(LLM, api_key="sk-1")}, "literal key"),
        ({"llm": {"provider": "openai_compatible", "model": "m"}}, "base_url"),
        ({"mjcf_path": "/nonexistent/scene.xml"}, "model not found"),
    ],
)
def test_a_bad_block_is_refused_naming_the_key(tmp_path, change, message):
    with pytest.raises(ValueError, match=message):
        generate_pose_composer_nodes(_config(tmp_path, **change))


def test_it_needs_the_x2_runtime_and_a_non_idle_skill_mode(tmp_path):
    config = _config(tmp_path)
    config["runtime"]["provider"] = "so101_robot"
    with pytest.raises(ValueError, match="aimdk_robot"):
        generate_pose_composer_nodes(config)
    config = _config(tmp_path)
    config["control_modes"]["named_motion"]["runtime_mode"] = "idle"
    with pytest.raises(ValueError, match="non-idle"):
        generate_pose_composer_nodes(config)
