import json
from pathlib import Path

import pytest

from embodied_bringup.launch_builders.embodied import generate_embodied_nodes
from robot_config.loader import load_robot_config_dict


def _normalize_launch_param_mapping(raw_params):
    normalized = {}
    for raw_key, raw_value in raw_params.items():
        key = "".join(getattr(item, "text", str(item)) for item in raw_key)
        if isinstance(raw_value, tuple):
            value = "".join(getattr(item, "text", str(item)) for item in raw_value)
        else:
            value = raw_value
        normalized[key] = value
    return normalized


def _decode_launch_json_string(raw_value: str):
    normalized = raw_value.strip()
    if len(normalized) >= 2 and normalized[0] == normalized[-1] and normalized[0] in {"'", '"'}:
        normalized = normalized[1:-1]
    return json.loads(normalized)


def test_task_entry_launch_params_do_not_include_unused_routing_config():
    robot_config = {
        "embodied": {
            "enabled": True,
            "execution": {},
            "entry": {},
            "named_poses": {},
            "named_targets": {},
            "skill_templates": {},
            "safety": {},
            "planner": {},
            "perception": {},
        }
    }

    nodes = generate_embodied_nodes(robot_config, active_control_mode="moveit_planning")
    task_entry = next(node for node in nodes if vars(node)["_Node__node_name"] == "task_entry_node")
    params = {}
    for group in vars(task_entry)["_Node__parameters"]:
        if isinstance(group, dict):
            for key, value in group.items():
                normalized_key = key[0].text if isinstance(key, tuple) else key.text
                normalized_value = value[0] if isinstance(value, tuple) else value
                if hasattr(normalized_value, "text"):
                    normalized_value = normalized_value.text
                params[normalized_key] = normalized_value

    assert "forward_unmatched_to_planner" not in params
    assert "reject_invalid_only" not in params
    assert "direct_skill_whitelist_json" not in params
    assert "planner_route_keywords_json" not in params


@pytest.mark.parametrize("config_name", ["so101_single_arm"])
def test_generate_embodied_nodes_passes_arm_joint_metadata(config_name):
    config_path = Path(__file__).parents[2] / "robot_config" / "config" / "robots" / f"{config_name}.yaml"

    if not config_path.exists():
        pytest.skip(f"Config file not found: {config_path}")

    config = load_robot_config_dict(config_path)
    config["embodied"]["enabled"] = True
    nodes = generate_embodied_nodes(config, "moveit_planning")

    skill_executor = next(
        node
        for node in nodes
        if node.__dict__.get("_Node__package") == "skill_library"
        and node.__dict__.get("_Node__node_executable") == "skill_executor_node"
    )
    params = _normalize_launch_param_mapping(skill_executor._Node__parameters[0])

    assert _decode_launch_json_string(params["arm_joint_names_json"]) == ["1", "2", "3", "4", "5"]
    assert set(_decode_launch_json_string(params["joint_limits_json"]).keys()) >= {"1", "2", "3", "4", "5"}


def test_launch_injects_only_rule_entry_skill_aliases():
    config_path = Path(__file__).parents[2] / "robot_config" / "config" / "robots" / "so101_single_arm.yaml"
    config = load_robot_config_dict(config_path)
    config["embodied"]["enabled"] = True
    nodes = generate_embodied_nodes(config, "moveit_planning")
    task_entry = next(node for node in nodes if vars(node)["_Node__node_name"] == "task_entry_node")
    params = _normalize_launch_param_mapping(task_entry._Node__parameters[0])
    aliases = _decode_launch_json_string(params["skill_aliases_json"])

    assert set(aliases) == {
        "dance_basic",
        "wave_hello",
        "nod_yes",
        "shake_no",
        "celebrate",
        "greet_observe_raise",
        "act_cute",
        "happy_spin_upright",
    }


def test_no_interaction_skills_node_is_generated():
    robot_config = {
        "embodied": {
            "enabled": True,
            "execution": {},
            "entry": {"visual_games": {"sorting_hat": {"enabled": True, "trigger_aliases": ["分院帽"]}}},
            "planner": {},
            "perception": {"enabled": True},
        }
    }

    nodes = generate_embodied_nodes(robot_config, active_control_mode="moveit_planning")

    node_names = [vars(node)["_Node__node_name"] for node in nodes]
    assert "interaction_skills_node" not in node_names

    task_entry = next(node for node in nodes if vars(node)["_Node__node_name"] == "task_entry_node")
    params = _normalize_launch_param_mapping(task_entry._Node__parameters[0])
    assert "perception_request_topic" in params
    assert "entry_visual_games_json" in params
    games = _decode_launch_json_string(params["entry_visual_games_json"])
    assert games["sorting_hat"]["enabled"] is True
    assert params["perception_enabled"] is True


class _FakeLaunchContext:
    def __init__(self, launch_configurations):
        self.launch_configurations = launch_configurations


def _load_launch_module():
    """Load the ``.launch.py`` file by path (not an importable package module)."""
    import importlib.util

    launch_path = Path(__file__).parents[1] / "launch" / "embodied_pipeline.launch.py"
    spec = importlib.util.spec_from_file_location("embodied_pipeline_launch", launch_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_launch_setup_aborts_when_game_enabled_but_perception_disabled():
    """with_perception:=false while a game is enabled must fail the launch, not
    start a node graph that routes the game to a dead topic. The validation gate
    runs before any base-launch include, so this raises without needing ROS
    share dirs."""
    module = _load_launch_module()

    fake_config = {
        "embodied": {
            "enabled": True,
            "perception": {"enabled": True},
            "entry": {"visual_games": {"sorting_hat": {"enabled": True, "trigger_aliases": ["分院帽"]}}},
        }
    }
    module._load_config = lambda *args, **kwargs: fake_config

    context = _FakeLaunchContext(
        {
            "robot_config": "so101_single_arm",
            "with_embodied": "true",
            "with_perception": "false",
        }
    )

    with pytest.raises(RuntimeError, match="visual_games"):
        module.launch_setup(context)


def test_handeye_grasp_config_launches_pick_pipeline():
    config_path = Path(__file__).parents[2] / "robot_config" / "config" / "robots" / "so101_handeye_realsense_only.yaml"
    config = load_robot_config_dict(config_path)
    assert config["grasp_execution"]["planner_node"]["enable_source_gripper_tabletop_sweep"] is False
    assert config["grasp_execution"]["ik"]["worker_count"] == 4
    config["embodied"]["enabled"] = True
    nodes = generate_embodied_nodes(config, "moveit_planning")
    executables = {(node.__dict__.get("_Node__package"), node.__dict__.get("_Node__node_executable")) for node in nodes}
    assert ("manipulation_execution", "pick_executor_node") in executables
    assert ("manipulation_service", "grasp_planner_node") in executables
    assert ("manipulation_service", "grasp_verifier_node") in executables
    assert ("perception_service", "grounded_sam2_node") in executables


def test_handeye_grasp_launch_auto_starts_parallel_ik_workers(monkeypatch, tmp_path):
    module = _load_launch_module()
    monkeypatch.setattr(module, "get_package_share_directory", lambda _package: str(tmp_path))
    config = {
        "grasp_execution": {
            "enabled": True,
            "auto_start_dependencies": True,
            "ik": {
                "worker_count": 4,
                "worker_namespace_prefix": "/ik_worker",
                "auto_start_workers": True,
            },
        }
    }

    action = module._parallel_ik_worker_action(config, "false")

    assert action is not None
    assert action.__class__.__name__ == "IncludeLaunchDescription"
