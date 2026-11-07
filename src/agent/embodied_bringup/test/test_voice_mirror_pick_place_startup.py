"""Software launch/catalog wiring for the combined voice, mirror and pick/place entry.

Merge the real robot overlays and bind the simulated public runtime description.
The Gateway compiles the real catalog using a strict-valid fixture model identity;
no model inference, vendor runtime, camera or motion executor is started.
"""

import json
from pathlib import Path

import rclpy
import yaml
from rclpy.parameter import Parameter

from embodied_bringup.launch_builders.embodied import generate_embodied_nodes
from robot_config.fixed_poses import resolve_fixed_poses
from robot_config.grasp_execution_config import validate_grasp_execution_config
from robot_config.interface_binding import bind_robot_interfaces
from robot_config.loader import _load_robot_section
from robot_config.placement_execution_config import validate_placement_execution_config
from robot_config.task_poses import resolve_task_pose_file
from robot_runtime.interface_description import build_description
from robot_runtime.profile import load_profile
from skill_library.skill_executor_node import SkillExecutorNode

from .test_embodied_launch_builder import _decode_launch_string, _normalize_launch_param_mapping
from .test_real_grasp_config_startup import _write_graspgen_bundle

ROOT = Path(__file__).resolve().parents[4]
CONFIG = ROOT / "src/robot_config/config/robots/aimdk_x2_voice_mirror_grasp.yaml"


def test_voice_mirror_entry_launches_pick_place_and_compiles_all_skill_families(tmp_path):
    # Model availability is a separate deployment check. Merge the actual YAML
    # rather than calling the full artifact validator on this model-free host.
    path, config = _load_robot_section(CONFIG)
    resolve_task_pose_file(config, path)
    # The fixed pose library is a per-deployment artifact compiled on site
    # (test_fixed_pose_path covers its resolution). This wiring test binds a
    # minimal stand-in the same way the loader binds the real one.
    library = tmp_path / "commissioning/fixed_poses.json"
    library.parent.mkdir(parents=True)
    library.write_text(json.dumps({"poses": {"arms_up": {"user_message": "举起双臂", "phrases": ["举手"]}}}))
    config["pose_composer"]["fixed_library"] = str(library)
    resolve_fixed_poses(config, path)
    config["_config_path"] = str(path)
    assert not validate_grasp_execution_config(config["grasp_execution"])
    assert not validate_placement_execution_config(config["placement_execution"])
    assert config["grasp_execution"]["enabled"] and config["placement_execution"]["enabled"]
    services = {item["id"]: item for item in config["perception_services"]["services"]}
    for name in ("grasp_grounding", "grasp_segmentation", "grasp_generation"):
        assert services[name]["enabled"] and services[name]["required"]
    assert config["placement_execution"]["perception"]["box_threshold"] == 0.50
    assert config["grasp_execution"]["backend_options"]["head"]["namespace"] == "/x2_grasp_head"

    _write_graspgen_bundle(tmp_path / "models")
    config["grasp_execution"].update(model_bundle_path=str(tmp_path / "models/graspgen"), model_deployment="torch_cpu")
    profile = ROOT / "src/robots/aimdk/aimdk_robot/profiles" / f"{config['runtime']['profile']}.yaml"
    config = bind_robot_interfaces(config, build_description(load_profile(str(profile)), use_sim=True))
    nodes = generate_embodied_nodes(config, "model_inference", motion_authorized=False, include_perception=False)
    params = {
        vars(node).get("_Node__node_executable"): _normalize_launch_param_mapping(node._Node__parameters[0])
        for node in nodes
        if vars(node).get("_Node__node_executable")
        in {"skill_executor_node", "grasp_pipeline_node", "place_pipeline_node"}
    }
    assert set(params) == {"skill_executor_node", "grasp_pipeline_node", "place_pipeline_node"}
    gateway_params = {
        key: yaml.safe_load(value) if isinstance(value, str) else value
        for key, value in params["skill_executor_node"].items()
    }
    assert gateway_params["skill_catalog_profile"] == "aimdk_x2_voice_mirror_grasp"
    for field, executable in (
        ("grasp_execution_json", "grasp_pipeline_node"),
        ("placement_execution_json", "place_pipeline_node"),
    ):
        pipeline = json.loads(_decode_launch_string(params[executable][field]))
        assert json.loads(gateway_params[field]) == pipeline
    gateway_params["execution_store_path"] = str(tmp_path / "executions.sqlite3")
    gateway_params["skill_catalog_source_mode"] = "development"
    gateway_params["skill_catalog_source_root"] = str(ROOT / "src/agent/skill_catalog")
    rclpy.init()
    gateway = None
    try:
        gateway = SkillExecutorNode(
            parameter_overrides=[Parameter(key, value=value) for key, value in gateway_params.items()]
        )
        snapshot = gateway._compile_runtime_snapshot()
        visible = set(snapshot.planner_visible_skill_names)
        assert set(config["embodied"]["agent"]["test_allowlist"]) == visible
        assert {
            "pick_object",
            "place_in_container",
            "mirror_start",
            "mirror_stop",
            "wave_hand",
            "self_introduction",
        } <= visible
        assert snapshot.templates["pick_object"]["executor"] == "grasp_pipeline"
        assert snapshot.templates["place_in_container"]["executor"] == "placement_pipeline"
        assert snapshot.capability_view["pick_object"]["required_control_mode"] == "moveit_planning"
        assert snapshot.capability_view["mirror_start"]["required_control_mode"] == "model_inference"
        assert snapshot.capability_view["wave_hand"]["required_control_mode"] == "named_motion"
    finally:
        if gateway is not None:
            gateway.destroy_node()
        rclpy.shutdown()
