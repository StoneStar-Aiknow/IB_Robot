from pathlib import Path

import pytest
import yaml

from robot_config.loader import load_robot_config_dict

CONFIG = Path(__file__).parents[1] / "config" / "robots" / "so101_handeye_realsense_grasp.yaml"


def _write_config(tmp_path: Path, mutate) -> Path:
    data = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    mutate(data["robot"]["grasp_execution"])
    path = tmp_path / "robot.yaml"
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return path


def test_grasp_execution_config_accepts_repository_profile() -> None:
    config = load_robot_config_dict(CONFIG)
    grasp_execution = config["grasp_execution"]

    assert grasp_execution["timeout_sec"] == 240.0
    assert grasp_execution["verification"] == "required"
    assert grasp_execution["max_execution_attempts"] == 3
    assert grasp_execution["recover_after_close_failure"] is True
    assert grasp_execution["recover_after_retention_failure"] is True
    assert grasp_execution["planner"] == {
        "confidence_threshold": 0.30,
        "grasp_threshold": 0.20,
        "timeout_sec": 120.0,
        "debug_output_mode": "none",
    }
    assert grasp_execution["candidate_selection"]["selection_attempts"] == 12
    assert grasp_execution["candidate_selection"]["max_candidates"] == 80
    assert grasp_execution["ik"]["worker_count"] == 4
    assert grasp_execution["ik"]["worker_namespace_prefix"] == "/ik_worker"
    assert grasp_execution["target_gripper"]["ik_orientation_guard"]["joint5_abs_max"] == 2.0
    assert grasp_execution["target_gripper"]["fixed_finger_robust_gap"]["measurement_tolerance_m"] == 0.001
    assert grasp_execution["contact_realign"]["max_iterations"] == 2
    assert grasp_execution["target_geometry"]["tabletop_clearance_m"] == -0.020
    assert grasp_execution["joint_state_topic"] == "/joint_states"
    assert grasp_execution["camera"]["rgb_topic"] == "/camera/wrist/image_raw"
    assert grasp_execution["state_wait"] == {
        "enabled": False,
        "minimum_sec": 0.05,
        "stable_sec": 0.06,
        "joint_delta_rad": 0.004,
        "gripper_tolerance_rad": 0.05,
        "gripper_joint": "6",
    }
    assert grasp_execution["detect_service"] == "/perception/grasp/grounding_detect"
    assert grasp_execution["segment_service"] == "/perception/grasp/segment_detections"
    services = {service["id"]: service for service in config["perception_services"]["services"]}
    assert services["grasp_grounding"]["deployment"] == "ascend_310p"
    assert services["grasp_grounding"]["service_type"] == "ibrobot_msgs/srv/GroundingDetect"
    assert services["grasp_segmentation"]["deployment"] == "ascend_310p"
    assert services["grasp_segmentation"]["service_type"] == "ibrobot_msgs/srv/SegmentDetections"
    assert grasp_execution["planner_node"]["inference_backend"] == "ascend_local"
    assert grasp_execution["planner_node"]["ascend_local_manifest_path"] == "/root/graspgen_310p_bundle"
    assert grasp_execution["planner_node"]["startup_warmup"] is True
    assert not any(key.startswith("remote_310p_") for key in grasp_execution["planner_node"])
    wrist = next(peripheral for peripheral in config["peripherals"] if peripheral.get("name") == "wrist")
    assert wrist["direct_topic_remap"] is True
    assert wrist["enable_pointcloud"] is False
    assert "pointcloud" not in wrist["streams"]
    assert all(
        observation["key"] != "observation.pointcloud.wrist" for observation in config["contract"]["observations"]
    )
    assert wrist["transform"]["parent_frame"] == "gripper"
    assert any(abs(float(wrist["transform"][axis])) > 1e-9 for axis in ("x", "y", "z", "roll", "pitch", "yaw"))


def test_grasp_execution_config_rejects_unknown_nested_key(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        lambda config: config["target_gripper"]["fixed_finger_base_side"].update({"min_aligment_cos": 0.0}),
    )

    with pytest.raises(ValueError, match="min_aligment_cos"):
        load_robot_config_dict(path)


def test_grasp_execution_config_rejects_negative_margin_gain(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        lambda config: config["target_gripper"].update({"fixed_finger_margin_width_gain": -0.1}),
    )

    with pytest.raises(ValueError, match="fixed_finger_margin_width_gain"):
        load_robot_config_dict(path)


def test_grasp_execution_config_rejects_negative_robust_gap_measurement_tolerance(tmp_path: Path) -> None:
    path = _write_config(
        tmp_path,
        lambda config: config["target_gripper"]["fixed_finger_robust_gap"].update({"measurement_tolerance_m": -0.0001}),
    )

    with pytest.raises(ValueError, match="measurement_tolerance_m"):
        load_robot_config_dict(path)


def test_grasp_execution_config_rejects_excessive_worker_count(tmp_path: Path) -> None:
    path = _write_config(tmp_path, lambda config: config["ik"].update({"worker_count": 9}))

    with pytest.raises(ValueError, match="worker_count"):
        load_robot_config_dict(path)


def test_grasp_execution_config_rejects_negative_state_wait_tolerance(tmp_path: Path) -> None:
    path = _write_config(tmp_path, lambda config: config["state_wait"].update({"joint_delta_rad": -0.1}))

    with pytest.raises(ValueError, match="joint_delta_rad"):
        load_robot_config_dict(path)


def test_grasp_execution_config_accepts_local_ascend_full_pipeline(tmp_path: Path) -> None:
    def enable_ascend(config):
        config["planner_node"].update(
            {
                "inference_backend": "ascend_local",
                "ascend_local_manifest_path": "/root/models/graspgen",
                "ascend_local_deployment_name": "ascend",
                "ascend_local_device_id": 0,
                "ascend_local_random_seed": -1,
            }
        )

    path = _write_config(tmp_path, enable_ascend)
    config = load_robot_config_dict(path)["grasp_execution"]

    assert config["detect_service"] == "/perception/grasp/grounding_detect"
    assert config["segment_service"] == "/perception/grasp/segment_detections"
    assert config["planner_node"]["inference_backend"] == "ascend_local"


def test_grasp_execution_config_requires_local_ascend_graspgen_bundle(tmp_path: Path) -> None:
    path = _write_config(tmp_path, lambda config: config["planner_node"].update({"ascend_local_manifest_path": ""}))

    with pytest.raises(ValueError, match="ascend_local_manifest_path"):
        load_robot_config_dict(path)
