import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

WORKSPACE = Path(__file__).resolve().parents[3]
SCRIPT_PATH = WORKSPACE / "scripts" / "test_banana_handeye_pick.py"
sys.path.insert(0, str(SCRIPT_PATH.parent))
SPEC = importlib.util.spec_from_file_location("test_banana_handeye_pick_config_target", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
pick_script = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pick_script
SPEC.loader.exec_module(pick_script)

ROBOT_CONFIG = WORKSPACE / "src" / "robot_config" / "config" / "robots" / "so101_handeye_realsense_only.yaml"


def test_grasp_script_uses_robot_config_execution_defaults():
    args = pick_script.parse_args(["--robot-config", str(ROBOT_CONFIG)])

    assert args.confidence_threshold == 0.30
    assert args.grasp_threshold == 0.20
    assert args.min_point_count == 100
    assert args.final_lift == 0.050
    assert args.observe_speed == 0.25
    assert args.lift_speed == 0.02
    assert args.grasp_verification_probe_lift_speed == 0.02
    assert args.max_execution_attempts == 1
    assert args.final_joint5_max == 2.0
    assert args.recover_after_close_failure is True
    assert args.recover_after_retention_failure is True
    assert args.contact_realign_max_iterations == 2
    assert args.ik_worker_count == 4
    assert args.ik_worker_prefix == "/ik_worker"
    assert args.pick_diagnostics_settle_s == 0.05
    assert args.so101_tabletop_clearance == -0.020


def test_explicit_cli_values_override_robot_config_execution_defaults():
    args = pick_script.parse_args(
        [
            "--robot-config",
            str(ROBOT_CONFIG),
            "--confidence-threshold",
            "0.42",
            "--final-lift=0.12",
            "--lift-speed",
            "0.4",
            "--ik-worker-count",
            "0",
            "--contact-realign-max-iterations",
            "7",
            "--pick-diagnostics-settle-s",
            "0.9",
            "--no-contact-realign",
            "--no-pick-diagnostics",
            "--so101-tabletop-clearance",
            "0.01",
        ]
    )

    assert args.confidence_threshold == 0.42
    assert args.final_lift == 0.12
    assert args.lift_speed == 0.4
    assert args.ik_worker_count == 0
    assert args.contact_realign_max_iterations == 7
    assert args.pick_diagnostics_settle_s == 0.9
    assert args.contact_realign is False
    assert args.pick_diagnostics is False
    assert args.so101_tabletop_clearance == 0.01


def test_graspgen_candidate_limit_applies_after_cheap_geometry_filters():
    args = pick_script.parse_args(["--robot-config", str(ROBOT_CONFIG), "--max-candidates", "2"])
    client = pick_script.BananaHandeyePickClient.__new__(pick_script.BananaHandeyePickClient)
    client.args = args
    client.handeye_matrix = np.eye(4, dtype=np.float64)
    client.ik_worker_clients = []
    client.use_ik_fk_contact_compensation = False
    client.last_graspgen_debug_output_dir = None
    client.wait_ik_ready = lambda: None

    candidates = [
        SimpleNamespace(
            index=index,
            confidence=0.9,
            collision_free=True,
            target_width_m=0.02,
            target_width_quality=1.0,
            target_width_min_offset_m=-0.01,
            target_width_max_offset_m=0.01,
        )
        for index in range(6)
    ]
    client.request_graspgen_candidates = lambda _base_to_gripper_tf: candidates
    client.rank_graspgen_candidates = lambda returned, _base_to_gripper_tf: [
        (candidate.index, candidate, None, 1.0, 1.0) for candidate in returned
    ]

    def graspgen_to_base_pose(candidate, _base_to_gripper_tf):
        pose = np.eye(4, dtype=np.float64)
        pose[0, 3] = float(candidate.index)
        return pose, pose, pose, np.zeros(3), (0.0, 0.0, 0.0), "test", 0.02

    client.graspgen_to_base_pose = graspgen_to_base_pose
    client.graspgen_targets_from_pose = lambda t_base_ee, _t_base_graspgen: (
        (float(t_base_ee[0, 3]), 0.0, 0.12),
        (float(t_base_ee[0, 3]), 0.0, 0.04),
        (float(t_base_ee[0, 3]), 0.0, 0.09),
        (0.0, 0.0, 0.0, 1.0),
        0.1,
    )
    client.graspgen_contact_point_base = lambda _pose: (0.0, 0.0, 0.0)
    client.graspgen_contact_point_camera = lambda _pose: (0.0, 0.0, 0.0)
    client.target_width_extent_for_candidate = lambda _candidate: None
    client.fixed_finger_envelope_for_candidate = lambda *_args: None
    client.fixed_finger_base_side_for_candidate = lambda grasp, _quat, _extent: SimpleNamespace(
        alignment_cos=-1.0 if grasp[0] < 3.0 else 1.0,
        inward_offset_m=0.01,
    )
    client._so101_gripper_geometry_metrics_batch = lambda values: [(0.0, 0.0, 0.0) for _ in values]
    client._record_so101_table_plane_shadow_clearances = lambda _execution, _planning: None
    client._is_within_workspace = lambda _grasp, _radius: (True, "")
    client._graspgen_height_guard = lambda _approach, _grasp, _contact: (True, "")

    evaluated_indices = []

    def evaluate_candidate_ik_contexts(contexts):
        evaluated_indices.extend(int(context["index"]) for context in contexts)
        return [
            {
                "failed_reason": "",
                "ik_fk_predicted_contact": None,
                "ik_fk_contact_error": None,
                "ik_fk_contact_residual_x": None,
                "ik_fk_contact_residual_y": None,
                "ik_fk_contact_z_error": None,
                "ik_fk_predicted_grasp_mesh_min_z": None,
                "ik_fk_predicted_tabletop_clearance": None,
                "ik_grasp_joint5": None,
                "ik_joint5_retry_applied": False,
                "ik_original_joint5": None,
                "ik_fk_approach_axis_error_deg": None,
                "ik_fk_closing_axis_error_deg": None,
                "ik_fk_fixed_finger_base_side": None,
                "ik_grasp_joint_state": None,
            }
            for _context in contexts
        ]

    client._evaluate_candidate_ik_contexts = evaluate_candidate_ik_contexts
    client._write_execution_debug_outputs = lambda **_kwargs: None
    base_to_gripper_tf = SimpleNamespace(
        transform=SimpleNamespace(
            translation=SimpleNamespace(x=0.0, y=0.0, z=0.0),
            rotation=SimpleNamespace(x=0.0, y=0.0, z=0.0, w=1.0),
        )
    )

    accepted = client.select_graspgen_candidates(base_to_gripper_tf)

    assert evaluated_indices == [3, 4]
    assert [candidate["index"] for candidate in accepted] == [3, 4]
    deferred = [record for record in client.execution_debug_records if record["stage"] == "ik_deferred"]
    assert [record["index"] for record in deferred] == [5]
