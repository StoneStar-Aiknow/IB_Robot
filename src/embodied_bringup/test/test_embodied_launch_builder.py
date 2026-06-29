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
