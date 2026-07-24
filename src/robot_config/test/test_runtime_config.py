from pathlib import Path

import pytest
import yaml

from robot_config.runtime_config import synthesize_runtime_config


def test_runtime_hardware_overrides_do_not_shadow_capabilities():
    base = {
        "robot": {
            "name": "so101_handeye_realsense_only",
            "grasp_execution": {"enabled": True, "lift_distance_m": 0.05},
            "embodied": {"skill_templates": {"pick_object": {"executor": "grasp_pipeline"}}},
            "ros2_control": {"port": "/dev/ttyACM0"},
            "peripherals": [{"name": "wrist", "width": 1280}],
            "contract": {"rate_hz": 20},
            "teleoperation": {"enabled": False},
        }
    }
    runtime = {
        "robot": {
            "name": "so101_handeye_realsense_only",
            "grasp_execution": {"enabled": False, "lift_distance_m": 0.165},
            "ros2_control": {"port": "/dev/ttyACM1"},
            "peripherals": [{"name": "wrist", "width": 640}],
            "contract": {"rate_hz": 30},
            "teleoperation": {"enabled": True, "active_device": "so101_leader"},
        }
    }

    merged = synthesize_runtime_config(base, runtime)["robot"]

    assert merged["ros2_control"]["port"] == "/dev/ttyACM1"
    assert merged["peripherals"][0]["width"] == 640
    assert merged["contract"]["rate_hz"] == 30
    assert merged["teleoperation"]["active_device"] == "so101_leader"
    assert merged["grasp_execution"] == {"enabled": True, "lift_distance_m": 0.05}
    assert "pick_object" in merged["embodied"]["skill_templates"]


def test_runtime_config_rejects_a_different_robot():
    with pytest.raises(ValueError, match="robot name mismatch"):
        synthesize_runtime_config(
            {"robot": {"name": "so101"}},
            {"robot": {"name": "lekiwi"}},
        )


def test_handeye_grasp_defaults_match_supervised_runtime_profile():
    config_path = Path(__file__).parents[1] / "config/robots/so101_handeye_realsense_only.yaml"
    robot = yaml.safe_load(config_path.read_text(encoding="utf-8"))["robot"]
    grasp = robot["grasp_execution"]

    assert grasp["lift_distance_m"] == pytest.approx(0.05)
    assert grasp["planner"]["confidence_threshold"] == pytest.approx(0.30)
    assert grasp["planner"]["grasp_threshold"] == pytest.approx(0.20)
    assert grasp["candidate_selection"]["min_contact_z"] == pytest.approx(-0.045)
    assert grasp["candidate_selection"]["min_point_count"] == 100
    assert grasp["max_execution_attempts"] == 1
    assert grasp["probe_lift_velocity_scaling"] == pytest.approx(0.02)
    assert grasp["lift_velocity_scaling"] == pytest.approx(0.02)
    assert grasp["ik"]["group_name"] == "arm"
    assert grasp["ik"]["timeout_sec"] == pytest.approx(0.20)
    assert grasp["ik"]["avoid_collisions"] is False
    assert grasp["ik"]["check_orientation"] is False
    assert grasp["ik"]["worker_count"] == 4
    assert grasp["ik"]["worker_namespace_prefix"] == "/ik_worker"
    assert grasp["ik"]["auto_start_workers"] is True
    assert grasp["contact_compensation"]["max_z_error_m"] == pytest.approx(0.020)
    assert grasp["prepared_candidate_scoring"]["enabled"] is True
    assert grasp["prepared_candidate_scoring"]["fixed_finger_envelope_weight"] == pytest.approx(0.55)
    assert grasp["prepared_candidate_scoring"]["centroid_distance_weight"] == pytest.approx(0.50)
    assert grasp["prepared_candidate_scoring"]["centroid_distance_scale_m"] == pytest.approx(0.010)
    assert grasp["prepared_candidate_scoring"]["reliable_max_opening_m"] == pytest.approx(0.072)
    base_side = grasp["target_gripper"]["fixed_finger_base_side"]
    assert base_side["enabled"] is True
    assert base_side["reference_point_base"] == [0.0, 0.0, 0.0]
    assert base_side["min_alignment_cos"] == pytest.approx(0.0)
    robust_gap = grasp["target_gripper"]["fixed_finger_robust_gap"]
    assert robust_gap["enabled"] is True
    assert robust_gap["max_target_gap_deficit_m"] == pytest.approx(0.003)
    assert grasp["target_gripper"]["ik_orientation_guard"]["joint5_abs_max"] == pytest.approx(2.0)
    assert grasp["contact_realign"]["pregrasp_clearance_m"] == pytest.approx(0.020)
    assert grasp["target_geometry"]["tabletop_clearance_m"] == pytest.approx(-0.020)
    assert "max_closing_axis_error_deg" not in grasp["target_geometry"]
    assert robot["embodied"]["named_poses"]["observe_table"]["position"] == {
        "x": 0.10,
        "y": -0.16,
        "z": 0.22,
    }
