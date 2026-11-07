"""Local files become a validated, digest-bound snapshot in the canonical loader."""

import copy
from pathlib import Path

import pytest
import yaml

from robot_config.loader import load_robot_config_dict, robot_config_digest
from robot_runtime.task_poses import normalize_task_poses

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "config/task_poses/x2_example.yaml"


@pytest.mark.parametrize(
    "robot_file",
    ["aimdk_x2_grasp_task_poses_310p.yaml", "aimdk_x2_voice_mirror_grasp.yaml"],
)
def test_grasp_ready_uses_table_side_pose(robot_file):
    from robot_config.loader import _load_robot_section
    from robot_config.task_poses import resolve_task_pose_file

    expected = {
        "left_shoulder_pitch_joint": -0.0486,
        "left_shoulder_roll_joint": 0.4486,
        "left_shoulder_yaw_joint": 0.3506,
        "left_elbow_joint": -1.45,
        "left_wrist_yaw_joint": 0.0038,
        "left_wrist_pitch_joint": 0.0678,
        "left_wrist_roll_joint": -0.0002,
        "right_shoulder_pitch_joint": -0.0554,
        "right_shoulder_roll_joint": -0.4486,
        "right_shoulder_yaw_joint": -0.3478,
        "right_elbow_joint": -1.45,
        "right_wrist_yaw_joint": -0.0029,
        "right_wrist_pitch_joint": 0.0678,
        "right_wrist_roll_joint": 0.0001,
    }
    path, config = _load_robot_section(ROOT / "config/robots" / robot_file)
    resolve_task_pose_file(config, path)
    task_poses = config["task_poses"]
    pose = task_poses["poses"]["grasp_ready"]
    assert set(task_poses["joint_names"]) == set(expected)
    assert pose["joint_positions"] == pytest.approx(expected, abs=1e-9)
    assert task_poses["skill_entry_poses"] == {"pick_object": "grasp_ready"}
    assert pose["hands"] == {"left": "open", "right": "open"}
    assert pose["hardware_approved"] is False
    assert pose["unloaded_operation_approved"] is False
    assert "HOME" not in pose["aliases"]
    assert "双臂放下" not in pose["aliases"]
    rest = task_poses["poses"]["arms_rest"]
    assert rest["joint_positions"] != pose["joint_positions"]
    assert "双手放下" in rest["aliases"]
    assert rest["hands"] == {"left": "open", "right": "open"}
    assert rest["hardware_approved"] is False
    assert rest["unloaded_operation_approved"] is False

    # Both entry points must stay within the grasp runtime's declared MC limits.
    _, grasp_config = _load_robot_section(ROOT / "config/robots/aimdk_x2_grasp.yaml")
    for target in (pose, rest):
        for name, position in target["joint_positions"].items():
            lower, upper = grasp_config["arm_limits"][name]
            assert lower <= position <= upper, name

    # Accepted tracking error must not take a resting shoulder across its MC limit.
    for name in ("left_shoulder_roll_joint", "right_shoulder_roll_joint"):
        lower, upper = grasp_config["arm_limits"][name]
        position = rest["joint_positions"][name]
        assert min(position - lower, upper - position) > rest["position_tolerance_rad"]


def test_task_pose_file_changes_execution_identity_and_stays_frozen(tmp_path):
    source = yaml.safe_load(EXAMPLE.read_text())
    file = tmp_path / "poses.yaml"
    file.write_text(yaml.safe_dump(source))
    robot = {
        "robot": {
            "name": "x2_task_test",
            "type": "x2_ultra_omnihand",
            "pose_composer": {"enabled": True},
            "task_poses": {"file": "poses.yaml"},
        }
    }
    config = tmp_path / "robot.yaml"
    config.write_text(yaml.safe_dump(robot))
    # Use the real resolver without requiring unrelated complete robot fields.
    from robot_config.task_poses import resolve_task_pose_file

    first = copy.deepcopy(robot["robot"])
    resolve_task_pose_file(first, config)
    digest = robot_config_digest(first)
    source["poses"]["grasp_ready"]["joint_positions"]["left_elbow_joint"] -= 0.01
    file.write_text(yaml.safe_dump(source))
    second = copy.deepcopy(robot["robot"])
    resolve_task_pose_file(second, config)
    assert robot_config_digest(first) == digest
    assert robot_config_digest(second) != digest


def test_canonical_loader_reads_file_before_validation(tmp_path):
    config = tmp_path / "robot.yaml"
    config.write_text("robot:\n  name: test_x2\n  type: x2_ultra_omnihand\n  task_poses: {file: absent.yaml}\n")
    with pytest.raises(FileNotFoundError):
        load_robot_config_dict(config)


def test_materialized_task_poses_reload_without_source_and_reject_tampering(tmp_path):
    from robot_config.task_poses import resolve_task_pose_file

    source = tmp_path / "poses.yaml"
    source.write_text(EXAMPLE.read_text())
    config = {"type": "x2_ultra_omnihand", "pose_composer": {"enabled": True}, "task_poses": {"file": str(source)}}
    resolve_task_pose_file(config, tmp_path / "robot.yaml")
    frozen = copy.deepcopy(config)
    consumer = yaml.safe_load(yaml.safe_dump(config))
    source.unlink()
    resolve_task_pose_file(consumer, tmp_path / "consumer.yaml")
    assert consumer == frozen
    consumer["task_poses"]["poses"]["grasp_ready"]["hands"].pop("right")
    with pytest.raises(ValueError, match="digest mismatch"):
        resolve_task_pose_file(consumer, tmp_path / "consumer.yaml")


def test_inline_task_poses_without_digest_are_not_a_consumer_snapshot(tmp_path):
    from robot_config.task_poses import resolve_task_pose_file

    config = {
        "type": "x2_ultra_omnihand",
        "pose_composer": {"enabled": True},
        "task_poses": yaml.safe_load(EXAMPLE.read_text()),
    }
    with pytest.raises(ValueError, match="digest-bound snapshot"):
        resolve_task_pose_file(config, tmp_path / "consumer.yaml")


@pytest.mark.parametrize(
    "fault", ["nan", "missing_joint", "duplicate_joint", "unknown_pose", "unknown_key", "bad_tolerance"]
)
def test_invalid_task_pose_files_are_rejected(fault):
    data = yaml.safe_load(EXAMPLE.read_text())
    pose = data["poses"]["grasp_ready"]
    if fault == "nan":
        pose["joint_positions"][data["joint_names"][0]] = float("nan")
    elif fault == "missing_joint":
        pose["joint_positions"].pop(data["joint_names"][0])
    elif fault == "duplicate_joint":
        data["joint_names"].append(data["joint_names"][0])
    elif fault == "unknown_pose":
        data["skill_entry_poses"]["pick_object"] = "missing"
    elif fault == "unknown_key":
        pose["typo"] = True
    else:
        pose["position_tolerance_rad"] = 100
    with pytest.raises(ValueError):
        normalize_task_poses(data)


def test_voice_mirror_grasp_resolves_task_poses_without_losing_deployment_features():
    from robot_config.loader import _load_robot_section
    from robot_config.task_poses import resolve_task_pose_file

    path, config = _load_robot_section(ROOT / "config/robots/aimdk_x2_voice_mirror_grasp.yaml")
    assert config["task_poses"]["file"] == "../task_poses/x2_example.yaml"
    resolve_task_pose_file(config, path)
    assert config["name"] == "aimdk_x2_voice_mirror_grasp"
    assert config["task_poses"]["skill_entry_poses"] == {"pick_object": "grasp_ready"}
    pose = config["task_poses"]["poses"]["grasp_ready"]
    assert pose["hardware_approved"] is False
    assert pose["unloaded_operation_approved"] is False
    assert pose["hands"] == {"left": "open", "right": "open"}
    assert config["pose_composer"]["enabled"] is True
    assert config["pose_composer"]["memory"]["enabled"] is False
    assert config["embodied"]["skill_catalog_profile"] == "aimdk_x2_voice_mirror_grasp"
    assert set(config["embodied"]["agent"]["test_allowlist"]) == {
        "wave_hand",
        "handshake",
        "raise_hand",
        "blow_kiss",
        "clap_hands",
        "mirror_start",
        "mirror_stop",
        "self_introduction",
        "pick_object",
        "free_pose",
        "return_to_task_pose",
    }
    assert config["voice_asr"]["enabled"] is True
    assert config["mirror_session"]["enabled"] is True
    assert config["mirror_session"]["voice"]["enabled"] is False
    assert config["mirror_session"]["operator"]["enabled"] is False
    assert config["grasp_execution"]["enabled"] is True
    assert set(config["skill_supported_control_modes"]) == {"named_motion", "model_inference", "moveit_planning"}
    assert all(
        config["control_modes"][mode]["runtime_mode"] == "policy_stream" for mode in ("named_motion", "moveit_planning")
    )
    services = {service["id"] for service in config["perception_services"]["services"]}
    assert {"grasp_grounding", "grasp_segmentation", "grasp_generation"} <= services


def test_opt_in_x2_deployment_resolves_packaged_pose_file():
    from robot_config.loader import _load_robot_section
    from robot_config.task_poses import resolve_task_pose_file

    path, config = _load_robot_section(ROOT / "config/robots/aimdk_x2_grasp_task_poses_310p.yaml")
    resolve_task_pose_file(config, path)
    assert config["task_poses"]["skill_entry_poses"] == {"pick_object": "grasp_ready"}
    assert not config["task_poses"]["poses"]["grasp_ready"]["hardware_approved"]
    assert config["embodied"]["skill_catalog_profile"] == "aimdk_x2_grasp_task_poses"


@pytest.mark.parametrize("hands", [None, {}, [], {"both": "open"}, {"left": "close"}, {"right": True}])
def test_invalid_hand_declarations_are_rejected(hands):
    data = yaml.safe_load(EXAMPLE.read_text())
    data["poses"]["grasp_ready"]["hands"] = hands
    with pytest.raises(ValueError, match="hands"):
        normalize_task_poses(data)


def test_hands_and_unloaded_approval_are_bound_to_snapshot_digest():
    data = yaml.safe_load(EXAMPLE.read_text())
    both = normalize_task_poses(data)
    pose = data["poses"]["grasp_ready"]
    pose["hands"] = {"left": "open"}
    left = normalize_task_poses(data)
    assert left["digest"] != both["digest"]
    pose["unloaded_operation_approved"] = True
    assert normalize_task_poses(data)["digest"] != left["digest"]
    both["poses"]["grasp_ready"]["hands"].pop("right")
    with pytest.raises(ValueError, match="digest mismatch"):
        normalize_task_poses(both)


def test_arm_only_normalization_does_not_add_hand_fields():
    data = yaml.safe_load(EXAMPLE.read_text())
    pose = data["poses"]["grasp_ready"]
    pose.pop("hands")
    pose.pop("unloaded_operation_approved")
    before = normalize_task_poses(data)
    assert "hands" not in before["poses"]["grasp_ready"]
    assert "unloaded_operation_approved" not in before["poses"]["grasp_ready"]
    assert normalize_task_poses(before) == before


@pytest.mark.parametrize("approval", ["true", 1, None])
def test_unloaded_approval_is_a_boolean(approval):
    data = yaml.safe_load(EXAMPLE.read_text())
    data["poses"]["grasp_ready"]["unloaded_operation_approved"] = approval
    with pytest.raises(ValueError, match="boolean"):
        normalize_task_poses(data)
