import json
from pathlib import Path

import pytest
from launch.actions import DeclareLaunchArgument, EmitEvent
from launch_ros.actions import Node

from embodied_bringup.launch_builders.embodied import _resolve_development_source_root, generate_embodied_nodes
from robot_config.launch_builders.perception_models import generate_perception_model_nodes
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


def _decode_launch_string(raw_value: str):
    normalized = raw_value.strip()
    if normalized.endswith("\n..."):
        normalized = normalized[:-4].rstrip()
    if len(normalized) >= 2 and normalized[0] == normalized[-1] and normalized[0] in {"'", '"'}:
        normalized = normalized[1:-1]
    return normalized


def _skill_executor_params(nodes):
    skill_executor = next(
        node
        for node in nodes
        if node.__dict__.get("_Node__package") == "skill_library"
        and node.__dict__.get("_Node__node_executable") == "skill_executor_node"
    )
    return _normalize_launch_param_mapping(skill_executor._Node__parameters[0])


def _normalize_launch_environment(node):
    return {
        "".join(getattr(item, "text", str(item)) for item in key): "".join(
            getattr(item, "text", str(item)) for item in value
        )
        for key, value in node.additional_env
    }


def test_hermes_entry_does_not_launch_voice_routing_node():
    robot_config = {
        "embodied": {
            "enabled": True,
            "entry_mode": "hermes",
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
    node_names = {vars(node)["_Node__node_name"] for node in nodes}
    assert "task_entry_node" not in node_names
    assert "task_executor_node" not in node_names


def test_launch_does_not_inject_legacy_skill_templates():
    embodied = {
        "enabled": True,
        "entry_mode": "hermes",
        "execution": {},
        "entry": {},
        "named_poses": {},
        "named_targets": {},
        "safety": {},
        "planner": {},
        "perception": {},
    }
    nodes = generate_embodied_nodes({"embodied": embodied}, active_control_mode="moveit_planning")
    params = _skill_executor_params(nodes)
    assert "skill_templates_json" not in params


@pytest.mark.parametrize("config_name", ["so101_single_arm"])
def test_generate_embodied_nodes_passes_arm_joint_metadata(config_name):
    config_path = Path(__file__).parents[2] / "robot_config" / "config" / "robots" / f"{config_name}.yaml"

    if not config_path.exists():
        pytest.skip(f"Config file not found: {config_path}")

    config = load_robot_config_dict(config_path)
    config["embodied"]["enabled"] = True
    nodes = generate_embodied_nodes(config, "moveit_planning")

    params = _skill_executor_params(nodes)

    assert _decode_launch_json_string(params["arm_joint_names_json"]) == ["1", "2", "3", "4", "5"]
    assert set(_decode_launch_json_string(params["joint_limits_json"]).keys()) >= {"1", "2", "3", "4", "5"}


def test_launch_injects_gateway_startup_params_from_runtime_and_ssot():
    robot_config = {
        "name": "test_robot",
        "default_control_mode": "model_inference",
        "skill_required_control_mode": "moveit_planning",
        "embodied": {
            "enabled": True,
            "skill_gateway_status_service": "/test/gateway_status",
            "timeouts": {
                "task_budget_sec": 90.0,
                "default_skill_timeout_sec": 12.0,
                "robot_state_freshness_sec": 0.25,
                "scene_freshness_sec": 0.75,
                "model_idle_timeout_sec": 45.0,
                "rpc_timeout_sec": 2.0,
                "gripper_settle_sec": 0.8,
            },
        },
    }

    nodes = generate_embodied_nodes(
        robot_config,
        active_control_mode="moveit_runtime_override",
        motion_authorized=True,
    )
    params = _skill_executor_params(nodes)

    assert params["motion_authorized"] is True
    assert _decode_launch_string(params["active_control_mode"]) == "moveit_runtime_override"
    assert _decode_launch_string(params["skill_required_control_mode"]) == "moveit_planning"
    assert _decode_launch_string(params["skill_gateway_status_service"]) == "/test/gateway_status"
    assert _decode_launch_string(params["robot_name"]) == "test_robot"
    assert params["default_skill_timeout_sec"] == 12.0
    assert params["task_budget_sec"] == 90.0
    assert params["robot_state_freshness_sec"] == 0.25
    assert params["scene_freshness_sec"] == 0.75
    assert params["model_idle_timeout_sec"] == 45.0
    assert params["rpc_timeout_sec"] == 2.0
    assert params["gripper_settle_sec"] == 0.8


def test_installed_profile_falls_back_to_ament_skill_catalog_source():
    robot_config = {
        "_config_path": "/opt/ibrobot/install/robot_config/share/robot_config/config/robots/robot.yaml",
        "embodied": {
            "enabled": True,
            "skill_catalog_source_mode": "development",
            "skill_catalog_source_root": "src/skill_catalog",
        },
    }

    params = _skill_executor_params(generate_embodied_nodes(robot_config, "moveit_planning"))

    assert _decode_launch_string(params["skill_catalog_source_mode"]) == "installed"
    assert _decode_launch_string(params["skill_catalog_source_root"]) == ""


def test_direct_builder_defaults_gateway_motion_authorization_to_false():
    nodes = generate_embodied_nodes(
        {"embodied": {"enabled": True, "safety": {"motion_authorized": True}}},
        "moveit_planning",
    )

    params = _skill_executor_params(nodes)

    assert params["motion_authorized"] is False


def test_no_interaction_skills_node_is_generated():
    robot_config = {
        "embodied": {
            "enabled": True,
            "entry_mode": "hermes",
            "execution": {},
            "entry": {"visual_games": {"sorting_hat": {"enabled": True, "trigger_aliases": ["分院帽"]}}},
            "perception": {"enabled": True},
        }
    }

    nodes = generate_embodied_nodes(robot_config, active_control_mode="moveit_planning")

    node_names = [vars(node)["_Node__node_name"] for node in nodes]
    assert "interaction_skills_node" not in node_names

    node_names = {vars(node)["_Node__node_name"] for node in nodes}
    assert "task_entry_node" not in node_names
    assert "perception_service_node" in node_names


def test_hermes_entry_mode_omits_voice_pipeline_nodes():
    nodes = generate_embodied_nodes(
        {"embodied": {"enabled": True, "entry_mode": "hermes"}},
        active_control_mode="moveit_planning",
    )

    node_names = {vars(node)["_Node__node_name"] for node in nodes}
    assert {"safety_guard_node", "skill_executor_node", "agent_plan_node"} <= node_names
    assert {"task_entry_node", "task_executor_node"}.isdisjoint(node_names)


def test_development_catalog_resolves_from_source_module_when_config_is_installed():
    installed_config = Path(__file__).parents[3] / "install/share/robot_config/config/robots/robot.yaml"

    resolved = _resolve_development_source_root(installed_config, "src/skill_catalog")

    assert resolved == (Path(__file__).parents[3] / "src/skill_catalog").resolve()


def test_non_hermes_entry_mode_is_rejected():
    import pytest

    with pytest.raises(ValueError, match="entry_mode must be hermes"):
        generate_embodied_nodes(
            {"embodied": {"enabled": True, "entry_mode": "voice"}},
            active_control_mode="moveit_planning",
        )


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


def test_authorize_motion_launch_argument_defaults_to_false():
    module = _load_launch_module()
    description = module.generate_launch_description()
    authorize_motion = next(
        entity
        for entity in description.entities
        if isinstance(entity, DeclareLaunchArgument) and entity.name == "authorize_motion"
    )

    assert authorize_motion.default_value[0].text == "false"


def test_caller_policy_requires_security_and_motion_authorization(monkeypatch, tmp_path):
    module = _load_launch_module()
    module._load_config = lambda *_args, **_kwargs: {
        "default_control_mode": "moveit_planning",
        "embodied": {"enabled": True, "perception": {"enabled": False}},
    }
    monkeypatch.setenv("ROS_SECURITY_ENABLE", "true")
    monkeypatch.setenv("ROS_SECURITY_STRATEGY", "Enforce")
    monkeypatch.setenv("ROS_SECURITY_KEYSTORE", str(tmp_path))
    context = _FakeLaunchContext(
        {
            "robot_config": "so101_single_arm",
            "with_embodied": "true",
            "authorize_motion": "true",
            "enable_caller_policy": "true",
        }
    )
    monkeypatch.setattr(module, "get_package_share_directory", lambda package: "/tmp")

    actions = module.launch_setup(context)
    assert actions


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


@pytest.mark.parametrize(("authorize_motion", "expected"), [(None, False), ("true", True)])
def test_launch_setup_passes_operator_motion_authorization(monkeypatch, authorize_motion, expected):
    module = _load_launch_module()
    config = {
        "default_control_mode": "model_inference",
        "embodied": {"enabled": True, "perception": {"enabled": False}},
    }
    module._load_config = lambda *_args, **_kwargs: config
    monkeypatch.setattr(module, "get_package_share_directory", lambda package: f"/tmp/{package}")
    generated = []

    def capture_nodes(robot_config, active_control_mode, *, motion_authorized=False, caller_policy_enabled=False):
        del caller_policy_enabled
        generated.append((robot_config, active_control_mode, motion_authorized))
        return []

    monkeypatch.setattr(module, "generate_embodied_nodes", capture_nodes)
    launch_configurations = {
        "robot_config": "so101_single_arm",
        "control_mode": "moveit_runtime_override",
        "with_embodied": "true",
    }
    if authorize_motion is not None:
        launch_configurations["authorize_motion"] = authorize_motion
        launch_configurations["enable_caller_policy"] = "true"
        monkeypatch.setenv("ROS_SECURITY_ENABLE", "true")
        monkeypatch.setenv("ROS_SECURITY_STRATEGY", "Enforce")
        monkeypatch.setenv("ROS_SECURITY_KEYSTORE", "/tmp")

    module.launch_setup(_FakeLaunchContext(launch_configurations))

    assert len(generated) == 1
    assert generated[0][1:] == ("moveit_runtime_override", expected)


def test_handeye_grasp_config_launches_pick_pipeline():
    config_path = (
        Path(__file__).parents[2] / "robot_config" / "config" / "robots" / "so101_handeye_realsense_grasp.yaml"
    )
    config = load_robot_config_dict(config_path)
    assert config["grasp_execution"]["planner_node"]["enable_source_gripper_tabletop_sweep"] is False
    assert config["grasp_execution"]["ik"]["worker_count"] == 4
    config["embodied"]["enabled"] = True
    nodes = generate_embodied_nodes(config, "moveit_planning")
    executables = {(node.__dict__.get("_Node__package"), node.__dict__.get("_Node__node_executable")) for node in nodes}
    assert ("manipulation_execution", "pick_executor_node") in executables
    assert ("manipulation_service", "grasp_planner_node") in executables
    assert ("manipulation_service", "grasp_verifier_node") in executables
    assert ("perception_service", "grounded_sam2_node") not in executables

    model_nodes = generate_perception_model_nodes(config)
    assert {vars(node).get("_Node__node_name") for node in model_nodes} == {
        "grasp_grounding_detect",
        "grasp_segment_detections",
    }
    model_params = {
        vars(node).get("_Node__node_name"): _normalize_launch_param_mapping(node._Node__parameters[0])
        for node in model_nodes
    }
    assert _decode_launch_string(str(model_params["grasp_grounding_detect"]["service_type"])) == (
        "ibrobot_msgs/srv/GroundingDetect"
    )
    assert _decode_launch_string(str(model_params["grasp_grounding_detect"]["service_endpoint"])) == (
        "/perception/grasp/grounding_detect"
    )
    assert _decode_launch_string(str(model_params["grasp_segment_detections"]["service_type"])) == (
        "ibrobot_msgs/srv/SegmentDetections"
    )
    assert _decode_launch_string(str(model_params["grasp_segment_detections"]["service_endpoint"])) == (
        "/perception/grasp/segment_detections"
    )

    planner = next(node for node in nodes if vars(node).get("_Node__node_name") == "grasp_planner")
    planner_params = _normalize_launch_param_mapping(planner._Node__parameters[0])
    assert str(planner_params["inference_backend"]).splitlines()[0] == "ascend_local"
    assert str(planner_params["ascend_local_manifest_path"]).splitlines()[0] == "/root/graspgen_310p_bundle"
    assert planner_params["startup_warmup"] is True
    assert _decode_launch_string(str(planner_params["legacy_detect_service"])) == ("/grasp_planner/detect_and_segment")
    assert "remote_310p_host" not in planner_params
    assert "host_runtime" not in planner_params
    assert _normalize_launch_environment(planner) == {
        "GOMP_SPINCOUNT": "0",
        "OMP_DYNAMIC": "FALSE",
        "OMP_NUM_THREADS": "4",
        "OMP_WAIT_POLICY": "PASSIVE",
        "OPENBLAS_NUM_THREADS": "1",
    }

    pick_executor = next(node for node in nodes if vars(node).get("_Node__node_name") == "pick_executor_node")
    pick_executor_params = _normalize_launch_param_mapping(pick_executor._Node__parameters[0])
    home_joint_positions = _decode_launch_json_string(str(pick_executor_params["home_joint_positions_json"]))
    assert home_joint_positions["5"] == 0.0


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


def test_source_workspace_profile_uses_absolute_development_catalog_root():
    config_path = Path(__file__).parents[2] / "robot_config" / "config" / "robots" / "so101_single_arm.yaml"
    config = load_robot_config_dict(config_path)
    config["embodied"]["enabled"] = True

    params = _skill_executor_params(generate_embodied_nodes(config, active_control_mode="moveit_planning"))

    assert _decode_launch_string(params["skill_catalog_source_mode"]) == "development"
    source_root = Path(_decode_launch_string(params["skill_catalog_source_root"]))
    assert source_root.is_absolute()
    assert source_root == Path(__file__).parents[3] / "src" / "skill_catalog"
    assert _decode_launch_string(params["skill_catalog_profile"]) == "so101_single_arm"


def test_embodied_runtime_waits_for_required_controllers():
    module = _load_launch_module()
    config = {
        "controller_startup_timeout": {"hardware": 30.0},
        "control_modes": {
            "moveit_planning": {
                "controllers": [
                    "joint_state_broadcaster",
                    "arm_trajectory_controller",
                    "gripper_trajectory_controller",
                ]
            }
        },
    }

    waiter = module._controller_ready_waiter(config, "moveit_planning", use_sim=False, auto_start=True)

    assert isinstance(waiter, Node)
    assert waiter.node_package == "robot_config"
    assert waiter.node_executable == "wait_for_controllers"
    assert "arm_trajectory_controller" in [_decode_launch_string(str(arg)) for arg in waiter._Node__arguments]


def test_embodied_runtime_readiness_handler_starts_only_after_success():
    module = _load_launch_module()
    runtime_actions = [object(), object()]
    handler = module._start_runtime_after_controller_readiness(runtime_actions)

    assert handler(type("Event", (), {"returncode": 0})(), None) == runtime_actions
    failure_actions = handler(type("Event", (), {"returncode": 1})(), None)
    assert len(failure_actions) == 1
    assert isinstance(failure_actions[0], EmitEvent)
