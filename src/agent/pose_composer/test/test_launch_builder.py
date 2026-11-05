"""The pose composer joins a launch only when the robot config enables it, with no literal key."""

import pytest

from pose_composer.launch_builder import generate_pose_composer_nodes

LLM = {"provider": "openai_compatible", "base_url": "http://127.0.0.1:1/v1", "model": "m"}


@pytest.fixture(scope="session", autouse=True)
def configured():
    # These tests validate launch wiring and temporary paths, not a physics model.
    return None


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
        ({"lookup": {"enabled": "true"}}, "lookup.enabled"),
        ({"lookup": {"unexpected": True}}, "lookup"),
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


def test_the_memory_block_reaches_the_node(tmp_path):
    config = _config(tmp_path, memory={"enabled": True, "path": str(tmp_path / "m.sqlite3"), "seeds": False})  # fmt: skip
    (node,) = generate_pose_composer_nodes(config)
    assert node.node_executable == "pose_composer_node"


@pytest.mark.parametrize(
    ("memory", "message"),
    [
        ({"enabled": True, "surprise": 1}, "unknown keys"),
        ({"enabled": True, "feedback": {"good": ["对"]}}, "unknown keys"),
        ({"enabled": True, "max_poses": 0}, "max_poses"),
    ],
)
def test_a_bad_memory_block_is_refused(tmp_path, memory, message):
    with pytest.raises(ValueError, match=message):
        generate_pose_composer_nodes(_config(tmp_path, memory=memory))


@pytest.mark.parametrize("key", ["max_joint_delta_rad", "max_hand_displacement_m"])
@pytest.mark.parametrize("value", [-0.1, float("inf"), float("nan"), True])
def test_invalid_commissioning_limits_are_refused(tmp_path, key, value):
    config = _config(tmp_path)
    config["pose_composer"][key] = value
    with pytest.raises(ValueError, match=key):
        generate_pose_composer_nodes(config)


def test_hand_wiring_uses_bound_description_and_gateway_endpoint():
    import json

    from pose_composer.launch_builder import _hand_restore_parameters

    assert _hand_restore_parameters({}) == ""
    config = {"task_poses": {"poses": {"ready": {"hands": {"left": "open"}}}}}
    with pytest.raises(ValueError, match="bound runtime"):
        _hand_restore_parameters(config)
    config.update(
        runtime={"interface_description": {"interfaces": {"joint.hand_stream": {"endpoint": "/bound/hands"}}}},
        grasp_execution={"backend": "x2"},
        embodied={"skill_gateway_status_service": "/bound/gateway_status"},
    )
    bound = json.loads(_hand_restore_parameters(config))
    assert bound["gateway_status_service"] == "/bound/gateway_status"
    assert bound["interface_description"] == config["runtime"]["interface_description"]


@pytest.mark.parametrize("use_sim", [True, False])
def test_builder_uses_resolved_launch_target_not_stale_config(tmp_path, monkeypatch, use_sim):
    from types import SimpleNamespace

    from pose_composer import launch_builder

    captured = {}

    def generate(config, *, use_sim):
        captured.update(config=config, use_sim=use_sim)
        return []

    config = _config(tmp_path)
    config["use_sim"] = not use_sim
    monkeypatch.setattr(launch_builder, "generate_pose_composer_nodes", generate)
    launch_builder.build_pose_composer(SimpleNamespace(robot_config=config, use_sim=use_sim))
    assert captured == {"config": config, "use_sim": use_sim}
