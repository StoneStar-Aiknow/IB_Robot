"""The staged hybrid pick-sim binding assertions (R1/R2, revision 2).

The E2E runs the production pick pipeline on the staged hybrid overlay; this
test pins what the E2E depends on, on BOTH platforms: the staged result is
the hybrid stage with grasp_execution and embodied enabled and the CUDA
deployment, the R-S1 launch fixes are in the staged result (require_model,
task_dispatch interface bindings, the staged development catalog and its
profile), and the staged catalog compiles against the robot context.

These are pure staging/load assertions — no simulator runs here (the sim
runs live in test_mobile_pick_simulation.py).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from robot_config.loader import load_robot_config_dict
from test.pick_sim_staging import stage_pick_sim_config

E2E_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PLATFORMS = ["gazebo", "mujoco"]


def _staged(tmp_path: Path, sim_platform: str) -> Path:
    overlay, _catalog = stage_pick_sim_config(
        tmp_path / f"staged-{sim_platform}",
        sim_platform=sim_platform,
        e2e_package_root=E2E_PACKAGE_ROOT,
        execution_store_path=tmp_path / "executions.sqlite3",
    )
    return overlay


@pytest.mark.parametrize("sim_platform", PLATFORMS)
def test_staged_overlay_is_the_hybrid_stage_with_cuda_grasp(tmp_path: Path, sim_platform: str) -> None:
    config = load_robot_config_dict(_staged(tmp_path, sim_platform), nav_stage="hybrid", defer_interface_binding=True)
    assert config["nav_stage"] == "hybrid"
    assert config["default_control_mode"] == "base_navigation"
    assert config["skill_required_control_mode"] == "moveit_planning"
    # R2: the CUDA deployment, values from the shipped CUDA fixture
    assert config["grasp_execution"]["enabled"] is True
    assert config["grasp_execution"]["model_deployment"] == "torch_cuda"
    assert config["grasp_execution"]["segment_service"] == ""
    planner_node = config["grasp_execution"]["planner_node"]
    assert planner_node["inference_backend"] == "local_cuda"
    assert planner_node["device"] == "cuda"
    assert planner_node["local_deployment_name"] == "torch_cuda"
    services = config["perception_services"]["services"]
    assert [service["id"] for service in services] == ["grasp_grounding"]
    assert services[0]["deployment"] == "torch_cuda"
    assert services[0]["bundle_path"] == "models/grounded_sam2_swint_ogc"
    # R2 Gazebo note: the colour CameraInfo on both platforms
    assert config["grasp_execution"]["camera"]["camera_info_topic"] == "/camera/wrist/camera_info"
    assert config["simulation"]["scene"] == "nav_grasp_room"
    assert config["simulation"]["platform"] == sim_platform


@pytest.mark.parametrize("sim_platform", PLATFORMS)
def test_staged_overlay_carries_the_rs1_launch_fixes(tmp_path: Path, sim_platform: str) -> None:
    config = load_robot_config_dict(_staged(tmp_path, sim_platform), nav_stage="hybrid", defer_interface_binding=True)
    # R-S1 fix 1: the live-binding branch requires an interface consumer
    assert config["runtime"]["require_model"] is True
    # R-S1 fix 2: task_executor resolves its motion surfaces from logical IDs
    assert config["task_dispatch"] == {
        "move_to_pose_interface": "motion.move_to_pose",
        "gripper_trajectory_interface": "joint.trajectory_gripper",
        "joint_state_interface": "joint.state",
    }
    # the runtime contract carries the global FK only (revision-2 profile fix)
    if "fk_service" in config["grasp_execution"]:
        assert config["grasp_execution"]["fk_service"] == "/motion/compute_fk"
    # R-S1 fix 4: a per-run execution store, inside the embodied block
    store = config["embodied"]["execution_store_path"]
    assert store and store != ""
    # R1: the disabled consumers of interfaces the platforms do not simulate
    assert config["semantic_mapping"]["enabled"] is False
    assert config["speech_direction"]["enabled"] is False
    assert config["voice_tts"]["enabled"] is False
    assert config["voice_asr"]["enabled"] is False
    assert config["embodied"]["imitate_human_motion"]["enabled"] is False
    assert config["embodied"]["idle_behaviors"]["sound_orientation"]["enabled"] is False
    assert [p["name"] for p in config["peripherals"]] == ["wrist", "mid360"]
    # navigation on the sim scene's map with the pick-sim parameters (the
    # test-only final-approach levers)
    nav2 = config["navigation"]["nav2_bringup"]
    assert nav2["enabled"] is True
    assert nav2["map_file"].endswith("nav_grasp_room.yaml")
    assert nav2["params_file"].endswith("nav2_params_navgrasp_pick_sim.yaml")


@pytest.mark.parametrize("sim_platform", PLATFORMS)
def test_staged_catalog_compiles_with_the_robot_context(tmp_path: Path, sim_platform: str) -> None:
    from skill_catalog.source import DevelopmentStagingSkillSource
    from skill_catalog.validator import validate_profile

    overlay = _staged(tmp_path, sim_platform)
    staged_catalog = Path(
        yaml.safe_load(overlay.read_text(encoding="utf-8"))["robot"]["nav_stages"]["hybrid"]["embodied"][
            "skill_catalog_source_root"
        ]
    )
    source = DevelopmentStagingSkillSource(staged_catalog)
    release = source.resolve_active_release()
    assert release.source_release_digest
    packages = {package.name for package in source.discover_packages(release)}
    # the instead_use closure of pick_object (validator-enforced)
    assert packages == {"pick_object", "place_in_container", "open_gripper_skill", "close_gripper_skill"}
    profile = source.load_profile(release, "lekiwi_pick_sim_test")
    diagnostics = validate_profile(
        profile, profile_name="lekiwi_pick_sim_test", robot_name="lekiwi_handeye_realsense_grasp_lidar"
    )
    assert diagnostics == []
    assert {entry["name"] for entry in profile["enabled_skills"]} == packages
